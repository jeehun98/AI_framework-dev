// test_flashattn_pipeline.cu
// FlashAttention-like 4-stage tile pipeline + cp.async 실험 커널 (NO NVTX)

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <math_constants.h>

#define CHECK_CUDA(cmd)                                                          \
    do {                                                                         \
        cudaError_t e = (cmd);                                                   \
        if (e != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(e));                                      \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

constexpr int WARP_SIZE = 32;

// === warp-level reduce helpers ===
__inline__ __device__ float warp_allreduce_max(float v) {
    unsigned mask = 0xffffffffu;
    v = fmaxf(v, __shfl_xor_sync(mask, v, 16));
    v = fmaxf(v, __shfl_xor_sync(mask, v, 8));
    v = fmaxf(v, __shfl_xor_sync(mask, v, 4));
    v = fmaxf(v, __shfl_xor_sync(mask, v, 2));
    v = fmaxf(v, __shfl_xor_sync(mask, v, 1));
    return v;
}

__inline__ __device__ float warp_allreduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_xor_sync(mask, v, 16);
    v += __shfl_xor_sync(mask, v, 8);
    v += __shfl_xor_sync(mask, v, 4);
    v += __shfl_xor_sync(mask, v, 2);
    v += __shfl_xor_sync(mask, v, 1);
    return v;
}

#if __CUDA_ARCH__ >= 800
// 16B cp.async wrapper (cache at all levels)
__device__ __forceinline__ void cp_async_16B(void* smem_ptr, const void* gmem_ptr) {
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void cp_async_wait_all() {
    // wait for all prior groups
    asm volatile("cp.async.wait_group 0;\n" ::);
}
#endif

// ============================================================================
// FlashAttention-like fused kernel (single head, B=1)
//  - Q: [N, D]
//  - K: [N, D]
//  - V: [N, D]
//  - O: [N, D]
//  tile_n: tile size on sequence axis
//  STAGES: SMEM ring-buffer depth (예: 4)
// ============================================================================

template<int D, int TILE_N, int STAGES>
__global__ void flashattn_like_4stage_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int seq_len)
{
    static_assert(STAGES >= 2, "STAGES must be >= 2");
    static_assert(TILE_N == WARP_SIZE, "This kernel assumes TILE_N == 32 == WARP_SIZE");
    static_assert(D % 4 == 0, "D must be multiple of 4 for 16B cp.async");

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int row     = blockIdx.x;          // one block per query row

    if (row >= seq_len) return;

    const float* q_row = Q + row * D;

    // shared memory layout:
    // [ K_stage0 (TILE_N * D) | ... | K_stage(STAGES-1) | V_stage0 | ... ]
    extern __shared__ float smem[];
    float* smem_K = smem;
    float* smem_V = smem + STAGES * TILE_N * D;

    const int   tile_elems = TILE_N * D;
    const float scale      = 1.0f / sqrtf(float(D));

    // Preload Q row into registers
    float q_reg[D];
#pragma unroll
    for (int d = 0; d < D; ++d) {
        q_reg[d] = q_row[d];
    }

    // Streaming softmax state (lane 0 only)
    float m_i = -CUDART_INF_F;
    float l_i = 0.0f;
    float out_reg[D];
#pragma unroll
    for (int d = 0; d < D; ++d) {
        out_reg[d] = 0.0f;
    }

    const int num_tiles = (seq_len + TILE_N - 1) / TILE_N;

    // -----------------------------
    // Preload first min(STAGES, num_tiles) tiles into SMEM (동기 로드)
    // -----------------------------
    int preload_tiles = num_tiles < STAGES ? num_tiles : STAGES;

    for (int t = 0; t < preload_tiles; ++t) {
        int stage_idx   = t;
        int tile_start  = t * TILE_N;
        int k_idx       = tile_start + lane_id;

#pragma unroll
        for (int d = 0; d < D; ++d) {
            float val = 0.0f;
            if (k_idx < seq_len) {
                val = K[k_idx * D + d];
            }
            int smem_offset = stage_idx * tile_elems + lane_id * D + d;
            smem_K[smem_offset] = val;
        }

#pragma unroll
        for (int d = 0; d < D; ++d) {
            float val = 0.0f;
            if (k_idx < seq_len) {
                val = V[k_idx * D + d];
            }
            int smem_offset = stage_idx * tile_elems + lane_id * D + d;
            smem_V[smem_offset] = val;
        }
    }

    __syncthreads();

    // -----------------------------
    // Main tile loop (ring buffer + cp.async prefetch)
    // -----------------------------
    for (int tile = 0; tile < num_tiles; ++tile) {
        int stage_idx  = tile % STAGES;
        int tile_start = tile * TILE_N;
        int k_idx      = tile_start + lane_id;

#if __CUDA_ARCH__ >= 800
        // cp.async로 prefetch한 타일들이 준비됐는지 보장
        cp_async_wait_all();
        __syncthreads();
#else
        __syncthreads();
#endif

        // 1) QKᵀ
        float p_j = -CUDART_INF_F;
        if (k_idx < seq_len) {
            float dot = 0.0f;
#pragma unroll
            for (int d = 0; d < D; ++d) {
                int off = stage_idx * tile_elems + lane_id * D + d;
                float k_val = smem_K[off];
                dot += q_reg[d] * k_val;
            }
            p_j = dot * scale;
        }

        float m_tile = warp_allreduce_max(p_j);

        // 2) Softmax (tile-level)
        float e_j = 0.0f;
        if (k_idx < seq_len) {
            e_j = expf(p_j - m_tile);
        }
        float s_tile = warp_allreduce_sum(e_j);

        // 3) streaming softmax + V 축적
        float m_new = 0.0f;
        float l_new = 0.0f;
        float old_scale = 0.0f;
        float new_scale = 0.0f;

        if (lane_id == 0) {
            if (m_i == -CUDART_INF_F && l_i == 0.0f) {
                // 첫 타일
                m_new = m_tile;
                float exp_m_tile_minus_m_new = 1.0f;   // m_tile - m_new == 0
                l_new = s_tile * exp_m_tile_minus_m_new;
                old_scale = 0.0f;
                new_scale = (l_new > 0.0f) ? (exp_m_tile_minus_m_new / l_new) : 0.0f;
            } else {
                m_new = fmaxf(m_i, m_tile);
                float exp_m_i_minus_m_new    = (m_i == -CUDART_INF_F) ? 0.0f : expf(m_i - m_new);
                float exp_m_tile_minus_m_new = expf(m_tile - m_new);
                l_new = l_i * exp_m_i_minus_m_new + s_tile * exp_m_tile_minus_m_new;
                if (l_new > 0.0f) {
                    old_scale = (l_i * exp_m_i_minus_m_new) / l_new;
                    new_scale = (exp_m_tile_minus_m_new) / l_new;
                } else {
                    old_scale = 0.0f;
                    new_scale = 0.0f;
                }
            }
        }

        unsigned mask = 0xffffffffu;
        m_new     = __shfl_sync(mask, m_new, 0);
        l_new     = __shfl_sync(mask, l_new, 0);
        old_scale = __shfl_sync(mask, old_scale, 0);
        new_scale = __shfl_sync(mask, new_scale, 0);

        // sum_j e_j * v_j[d] (warp reduction)
        for (int d = 0; d < D; ++d) {
            float contrib = 0.0f;
            if (k_idx < seq_len) {
                int off = stage_idx * tile_elems + lane_id * D + d;
                float v_val = smem_V[off];
                contrib = e_j * v_val;
            }
            float sum_d = warp_allreduce_sum(contrib);

            if (lane_id == 0) {
                // out = out * old_scale + new_scale * sum_e_v
                out_reg[d] = out_reg[d] * old_scale + new_scale * sum_d;
            }
        }

        if (lane_id == 0) {
            m_i = m_new;
            l_i = l_new;
        }

        __syncwarp();

        // --------------------------------------------------
        // cp.async prefetch: tile + STAGES 를 같은 stage_idx 버퍼에 로드
        //   - 현재 타일은 이미 다 사용 완료 상태
        //   - N % TILE_N == 0 가정하면 tail 처리 신경쓸 필요 없음 (지금 N=1024 OK)
        // --------------------------------------------------
        int preload_tile = tile + STAGES;
        if (preload_tile < num_tiles) {
            int preload_stage = stage_idx;
            int preload_start = preload_tile * TILE_N;
            int pk_idx        = preload_start + lane_id;

#if __CUDA_ARCH__ >= 800
            // cp.async 로드 (16B 단위)
            const float* gmem_k = K + pk_idx * D;
            const float* gmem_v = V + pk_idx * D;

            float* smem_k_base = smem_K + preload_stage * tile_elems + lane_id * D;
            float* smem_v_base = smem_V + preload_stage * tile_elems + lane_id * D;

#pragma unroll
            for (int d = 0; d < D; d += 4) {
                void*       s_k = (void*)(smem_k_base + d);
                const void* g_k = (const void*)(gmem_k + d);
                cp_async_16B(s_k, g_k);

                void*       s_v = (void*)(smem_v_base + d);
                const void* g_v = (const void*)(gmem_v + d);
                cp_async_16B(s_v, g_v);
            }

            cp_async_commit_group();
#else
            // (구형 아키텍처용 fallback: 동기 로드)
            __syncthreads();
#pragma unroll
            for (int d = 0; d < D; ++d) {
                float valK = 0.0f;
                float valV = 0.0f;
                if (pk_idx < seq_len) {
                    valK = K[pk_idx * D + d];
                    valV = V[pk_idx * D + d];
                }
                int off_k = preload_stage * tile_elems + lane_id * D + d;
                int off_v = preload_stage * tile_elems + lane_id * D + d;
                smem_K[off_k] = valK;
                smem_V[off_v] = valV;
            }
            __syncthreads();
#endif
        }
    }

    // Store
    if (lane_id == 0) {
#pragma unroll
        for (int d = 0; d < D; ++d) {
            O[row * D + d] = out_reg[d];
        }
    }
}

// ============================================================================
// CPU reference attention (QKᵀ -> softmax -> softmax·V)
// ============================================================================

void cpu_attention_ref(const std::vector<float>& Q,
                       const std::vector<float>& K,
                       const std::vector<float>& V,
                       std::vector<float>& O,
                       int N, int D)
{
    const float scale = 1.0f / std::sqrt((float)D);

    std::vector<float> logits(N);
    std::vector<float> probs(N);

    for (int i = 0; i < N; ++i) {
        const float* q = &Q[i * D];

        // 1) logits = q · K^T
        float max_logit = -1e30f;
        for (int j = 0; j < N; ++j) {
            const float* k = &K[j * D];
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) {
                dot += q[d] * k[d];
            }
            float val = dot * scale;
            logits[j] = val;
            if (val > max_logit) max_logit = val;
        }

        // 2) softmax
        float sum_exp = 0.0f;
        for (int j = 0; j < N; ++j) {
            float e = std::exp(logits[j] - max_logit);
            probs[j] = e;
            sum_exp += e;
        }
        for (int j = 0; j < N; ++j) {
            probs[j] /= sum_exp;
        }

        // 3) O[i] = sum_j probs[j] * V[j]
        float* o = &O[i * D];
        for (int d = 0; d < D; ++d) o[d] = 0.0f;

        for (int j = 0; j < N; ++j) {
            const float* v = &V[j * D];
            float w = probs[j];
            for (int d = 0; d < D; ++d) {
                o[d] += w * v[d];
            }
        }
    }
}

// ============================================================================
// Driver
// ============================================================================

int main() {
    constexpr int D       = 64;
    constexpr int TILE_N  = 32;
    constexpr int STAGES  = 4;
    const int     N       = 1024;
    const int     NUM_RUN = 50;

    printf("FlashAttention-like tile pipeline test (cp.async, NO NVTX)\n");
    printf("N = %d, D = %d, TILE_N = %d, STAGES = %d\n",
           N, D, TILE_N, STAGES);

    size_t bytes_QKV = size_t(N) * D * sizeof(float);
    size_t bytes_O   = bytes_QKV;

    float *d_Q, *d_K, *d_V, *d_O;
    CHECK_CUDA(cudaMalloc(&d_Q, bytes_QKV));
    CHECK_CUDA(cudaMalloc(&d_K, bytes_QKV));
    CHECK_CUDA(cudaMalloc(&d_V, bytes_QKV));
    CHECK_CUDA(cudaMalloc(&d_O, bytes_O));

    // Host side init
    std::vector<float> h_Q(N * D), h_K(N * D), h_V(N * D);
    std::mt19937 rng(2025);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < N * D; ++i) {
        h_Q[i] = dist(rng);
        h_K[i] = dist(rng);
        h_V[i] = dist(rng);
    }

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), bytes_QKV, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), bytes_QKV, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), bytes_QKV, cudaMemcpyHostToDevice));

    dim3 block(WARP_SIZE, 1, 1);
    dim3 grid(N, 1, 1);

    size_t shmem_bytes = 2 * STAGES * TILE_N * D * sizeof(float);
    printf("Requested dynamic shared memory per block: %.2f KB\n",
           shmem_bytes / 1024.0f);

    // 디바이스 shared memory 한도 확인 및 opt-in 설정
    int dev = 0;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    printf("Device sharedMemPerBlock      : %.2f KB\n",
           prop.sharedMemPerBlock / 1024.0f);
    printf("Device sharedMemPerBlockOptin : %.2f KB\n",
           prop.sharedMemPerBlockOptin / 1024.0f);

    if (shmem_bytes > prop.sharedMemPerBlock) {
        if (shmem_bytes <= prop.sharedMemPerBlockOptin) {
            // opt-in to larger dynamic shared memory
            CHECK_CUDA(cudaFuncSetAttribute(
                flashattn_like_4stage_kernel<D, TILE_N, STAGES>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                (int)shmem_bytes));
            printf("Opt-in dynamic shared memory set to %.2f KB\n",
                   shmem_bytes / 1024.0f);
        } else {
            printf("ERROR: requested %.2f KB > opt-in limit %.2f KB\n",
                   shmem_bytes / 1024.0f,
                   prop.sharedMemPerBlockOptin / 1024.0f);
            return 1;
        }
    }

    printf("Shared memory per block (launch): %.2f KB\n",
           shmem_bytes / 1024.0f);

    // Warmup
    flashattn_like_4stage_kernel<D, TILE_N, STAGES><<<grid, block, shmem_bytes>>>(
        d_Q, d_K, d_V, d_O, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUN; ++i) {
        flashattn_like_4stage_kernel<D, TILE_N, STAGES><<<grid, block, shmem_bytes>>>(
            d_Q, d_K, d_V, d_O, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= NUM_RUN;

    printf("Kernel avg time (cp.async): %.6f ms (over %d runs)\n", ms, NUM_RUN);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    // GPU 결과 가져오기
    std::vector<float> h_O(N * D);
    CHECK_CUDA(cudaMemcpy(h_O.data(), d_O, bytes_O, cudaMemcpyDeviceToHost));

    // --- 출력 통계 ---
    double sum = 0.0;
    double sum_abs = 0.0;
    double max_abs = 0.0;

    for (int i = 0; i < N * D; ++i) {
        double v = h_O[i];
        sum     += v;
        sum_abs += std::fabs(v);
        max_abs = std::max(max_abs, std::fabs(v));
    }

    printf("O stats: mean = %.6e, mean|v| = %.6e, max|v| = %.6e\n",
           sum / (N * D), sum_abs / (N * D), max_abs);

    printf("O[0,0] = %.9f, O[N-1, D-1] = %.9f\n",
           h_O[0], h_O.back());

    // --- CPU reference와 비교 ---
    std::vector<float> h_O_ref(N * D);
    cpu_attention_ref(h_Q, h_K, h_V, h_O_ref, N, D);

    double max_diff = 0.0;
    for (int i = 0; i < N * D; ++i) {
        double diff = std::fabs((double)h_O[i] - (double)h_O_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Max diff (GPU vs CPU ref) = %.6e\n", max_diff);

    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O));

    return 0;
}
/*
ncu --set full --launch-skip 0 --launch-count 1 ./test_flashattn_pipeline

*/