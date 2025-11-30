// attention_fusion_test.cu
// naive attention (QK^T -> softmax -> softmax·V) vs fused attention (one kernel)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                      \
    do {                                                                      \
        cudaError_t _err = (expr);                                            \
        if (_err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_err));            \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// ------------------------------------------------------------
// Utils
// ------------------------------------------------------------

float rand_float() {
    return (float)rand() / RAND_MAX - 0.5f;  // [-0.5, 0.5]
}

void init_random(std::vector<float>& v) {
    for (auto& x : v) x = rand_float();
}

float max_abs_diff(const std::vector<float>& a,
                   const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

// ------------------------------------------------------------
// CPU reference attention
//   Q: [S, D]
//   K: [S, D]
//   V: [S, D]
//   O: [S, D]
// ------------------------------------------------------------

void attention_cpu_ref(const float* Q, const float* K, const float* V,
                       float* O, int S, int D) {
    const float scale = 1.0f / std::sqrt((float)D);

    std::vector<float> scores(S * S);
    std::vector<float> softmax(S * S);

    for (int i = 0; i < S; ++i) {
        // scores[i, j]
        for (int j = 0; j < S; ++j) {
            float s = 0.0f;
            for (int d = 0; d < D; ++d) {
                s += Q[i * D + d] * K[j * D + d];
            }
            scores[i * S + j] = s * scale;
        }

        // row-wise softmax
        float m = -FLT_MAX;
        for (int j = 0; j < S; ++j)
            m = std::max(m, scores[i * S + j]);

        float denom = 0.0f;
        for (int j = 0; j < S; ++j) {
            float e = std::exp(scores[i * S + j] - m);
            softmax[i * S + j] = e;
            denom += e;
        }
        for (int j = 0; j < S; ++j)
            softmax[i * S + j] /= denom;
    }

    // O = softmax * V
    for (int i = 0; i < S; ++i) {
        for (int d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (int j = 0; j < S; ++j) {
                acc += softmax[i * S + j] * V[j * D + d];
            }
            O[i * D + d] = acc;
        }
    }
}

// ------------------------------------------------------------
// Naive GPU kernels
// ------------------------------------------------------------

// 1) QK^T
// Q: [S, D], K: [S, D], Scores: [S, S] (row-major)
__global__ void naive_qk_kernel(const float* __restrict__ Q,
                                const float* __restrict__ K,
                                float* __restrict__ Scores,
                                int S, int D) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // query index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // key index

    if (row >= S || col >= S) return;

    const float scale = 1.0f / sqrtf((float)D);

    float acc = 0.0f;
    for (int d = 0; d < D; ++d) {
        acc += Q[row * D + d] * K[col * D + d];
    }
    Scores[row * S + col] = acc * scale;
}

// 2) row-wise softmax in-place on Scores [S, S]
__global__ void naive_softmax_kernel(float* __restrict__ Scores,
                                     int S) {
    // one block per row
    int row = blockIdx.x;
    if (row >= S) return;

    extern __shared__ float smem[];  // for reduction
    float* smax = smem;
    float* ssum = smem;  // reuse same buffer (two-phase)

    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    // phase 1: max
    float local_max = -FLT_MAX;
    for (int j = tid; j < S; j += stride) {
        float v = Scores[row * S + j];
        local_max = fmaxf(local_max, v);
    }

    smax[tid] = local_max;
    __syncthreads();

    // reduce max
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            smax[tid] = fmaxf(smax[tid], smax[tid + offset]);
        }
        __syncthreads();
    }
    float row_max = smax[0];

    // phase 2: exp & sum
    float local_sum = 0.0f;
    for (int j = tid; j < S; j += stride) {
        float e = expf(Scores[row * S + j] - row_max);
        Scores[row * S + j] = e;
        local_sum += e;
    }

    ssum[tid] = local_sum;
    __syncthreads();

    // reduce sum
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            ssum[tid] += ssum[tid + offset];
        }
        __syncthreads();
    }
    float denom = ssum[0];

    // phase 3: normalize
    for (int j = tid; j < S; j += stride) {
        Scores[row * S + j] /= denom;
    }
}

// 3) output = softmax * V
// scores: [S, S], V: [S, D], O: [S, D]
__global__ void naive_pv_kernel(const float* __restrict__ Scores,
                                const float* __restrict__ V,
                                float* __restrict__ O,
                                int S, int D) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // query
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // out dim

    if (row >= S || col >= D) return;

    float acc = 0.0f;
    for (int j = 0; j < S; ++j) {
        acc += Scores[row * S + j] * V[j * D + col];
    }
    O[row * D + col] = acc;
}

// ------------------------------------------------------------
// Fused attention kernel (FlashAttention-style in *graph* sense)
//
// - 하나의 kernel에서:
//   * QK^T score 계산
//   * row-wise softmax (online log-sum-exp 방식)
//   * softmax * V 까지 모두 처리
//
// - score matrix S[i, j]는 global에 만들지 않음.
// - 단순화를 위해:
//   * 각 block이 하나의 query row i 담당
//   * 각 thread가 output dim d 하나 담당
//   * online softmax streaming으로 score를 재사용
//
//   out[i, d] = Σ_j softmax_ij * V[j, d]
// ------------------------------------------------------------

__global__ void fused_attention_kernel(const float* __restrict__ Q,
                                       const float* __restrict__ K,
                                       const float* __restrict__ V,
                                       float* __restrict__ O,
                                       int S, int D) {
    int row = blockIdx.x;          // query index
    int d   = threadIdx.x;         // output dim index

    if (row >= S || d >= D) return;

    const float scale = 1.0f / sqrtf((float)D);

    // Q[row, :] 를 shared에 캐싱 (한 번만 global에서 읽기)
    __shared__ float s_Q[1024];    // D <= 1024 가정
    if (threadIdx.x < D) {
        s_Q[threadIdx.x] = Q[row * D + threadIdx.x];
    }

    // online softmax 상태 + broadcast용 shared 변수
    __shared__ float s_m;       // running max
    __shared__ float s_l;       // running sum of exp
    __shared__ float s_exp_m;   // exp_m
    __shared__ float s_exp_s;   // exp_s (현재 score의 exp 쪽)

    if (threadIdx.x == 0) {
        s_m = -FLT_MAX;
        s_l = 0.0f;
        s_exp_m = 0.0f;
        s_exp_s = 0.0f;
    }
    __syncthreads();

    float out_val = 0.0f;

    // 각 key j에 대해 streaming softmax + V accumulation
    for (int j = 0; j < S; ++j) {
        if (threadIdx.x == 0) {
            // score = dot(Q[row], K[j]) (Q는 shared, K는 global)
            float score = 0.0f;
            for (int k = 0; k < D; ++k) {
                score += s_Q[k] * K[j * D + k];
            }
            score *= scale;

            float m  = s_m;
            float l  = s_l;
            float new_m = fmaxf(m, score);

            float exp_m = (m == -FLT_MAX) ? 0.0f : expf(m - new_m);
            float exp_s = expf(score - new_m);

            s_m      = new_m;
            s_l      = l * exp_m + exp_s;
            s_exp_m  = exp_m;
            s_exp_s  = exp_s;
        }

        // 여기서 모든 thread가 s_exp_m, s_exp_s, s_l을 보기 전에 sync
        __syncthreads();

        // 각 thread는 자기 d에 대한 out_val 업데이트만 수행
        float v = V[j * D + d];
        out_val = out_val * s_exp_m + s_exp_s * v;

        // 다음 j iteration에서 thread 0이 shared 값을 덮어쓰기 전에
        // 모든 thread가 위 값을 다 사용하도록 sync
        __syncthreads();
    }

    float l = s_l;
    O[row * D + d] = out_val / l;
}


// ------------------------------------------------------------
// main
//   Usage: ./attention_fusion_test [S] [D] [iters]
// ------------------------------------------------------------

int main(int argc, char** argv) {
    int S = 512;   // sequence length
    int D = 64;    // head dim
    int iters = 100;

    if (argc >= 2) S = std::atoi(argv[1]);
    if (argc >= 3) D = std::atoi(argv[2]);
    if (argc >= 4) iters = std::atoi(argv[3]);

    printf("Attention size: S=%d, D=%d, iters=%d\n", S, D, iters);

    size_t bytes_Q = (size_t)S * D * sizeof(float);
    size_t bytes_K = bytes_Q;
    size_t bytes_V = bytes_Q;
    size_t bytes_scores = (size_t)S * S * sizeof(float);
    size_t bytes_O = bytes_Q;

    std::vector<float> h_Q(S * D);
    std::vector<float> h_K(S * D);
    std::vector<float> h_V(S * D);
    std::vector<float> h_O_ref(S * D);
    std::vector<float> h_O_naive(S * D);
    std::vector<float> h_O_fused(S * D);

    srand(0);
    init_random(h_Q);
    init_random(h_K);
    init_random(h_V);

    // CPU reference
    printf("Computing CPU reference...\n");
    attention_cpu_ref(h_Q.data(), h_K.data(), h_V.data(), h_O_ref.data(), S, D);

    float *d_Q, *d_K, *d_V;
    float *d_scores;
    float *d_O_naive, *d_O_fused;

    CUDA_CHECK(cudaMalloc(&d_Q, bytes_Q));
    CUDA_CHECK(cudaMalloc(&d_K, bytes_K));
    CUDA_CHECK(cudaMalloc(&d_V, bytes_V));
    CUDA_CHECK(cudaMalloc(&d_scores, bytes_scores));
    CUDA_CHECK(cudaMalloc(&d_O_naive, bytes_O));
    CUDA_CHECK(cudaMalloc(&d_O_fused, bytes_O));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), bytes_Q, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), bytes_K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), bytes_V, cudaMemcpyHostToDevice));

    // --------------------------------------------------------
    // Warmup & correctness check: naive pipeline
    // --------------------------------------------------------

    dim3 block_qk(16, 16);
    dim3 grid_qk((S + block_qk.x - 1) / block_qk.x,
                 (S + block_qk.y - 1) / block_qk.y);

    dim3 block_softmax(256);
    dim3 grid_softmax(S);
    size_t smem_softmax = block_softmax.x * sizeof(float);

    dim3 block_pv(16, 16);
    dim3 grid_pv((D + block_pv.x - 1) / block_pv.x,
                 (S + block_pv.y - 1) / block_pv.y);

    // one run
    naive_qk_kernel<<<grid_qk, block_qk>>>(d_Q, d_K, d_scores, S, D);
    CUDA_CHECK(cudaGetLastError());

    naive_softmax_kernel<<<grid_softmax, block_softmax, smem_softmax>>>(d_scores, S);
    CUDA_CHECK(cudaGetLastError());

    naive_pv_kernel<<<grid_pv, block_pv>>>(d_scores, d_V, d_O_naive, S, D);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_O_naive.data(), d_O_naive, bytes_O, cudaMemcpyDeviceToHost));

    float diff_naive = max_abs_diff(h_O_ref, h_O_naive);
    printf("Max abs diff (ref vs naive) = %.6e\n", diff_naive);

    // --------------------------------------------------------
    // Warmup & correctness: fused kernel
    // --------------------------------------------------------

    dim3 block_fused(D);   // one thread per output dim
    dim3 grid_fused(S);    // one block per query row

    fused_attention_kernel<<<grid_fused, block_fused>>>(d_Q, d_K, d_V,
                                                        d_O_fused, S, D);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_O_fused.data(), d_O_fused, bytes_O, cudaMemcpyDeviceToHost));

    float diff_fused = max_abs_diff(h_O_ref, h_O_fused);
    printf("Max abs diff (ref vs fused)  = %.6e\n", diff_fused);

    // --------------------------------------------------------
    // Timing: naive pipeline
    // --------------------------------------------------------

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));

    for (int it = 0; it < iters; ++it) {
        naive_qk_kernel<<<grid_qk, block_qk>>>(d_Q, d_K, d_scores, S, D);
        naive_softmax_kernel<<<grid_softmax, block_softmax, smem_softmax>>>(d_scores, S);
        naive_pv_kernel<<<grid_pv, block_pv>>>(d_scores, d_V, d_O_naive, S, D);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_naive = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));
    ms_naive /= iters;

    // --------------------------------------------------------
    // Timing: fused kernel
    // --------------------------------------------------------

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));

    for (int it = 0; it < iters; ++it) {
        fused_attention_kernel<<<grid_fused, block_fused>>>(d_Q, d_K, d_V,
                                                            d_O_fused, S, D);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_fused = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_fused, start, stop));
    ms_fused /= iters;

    // --------------------------------------------------------
    // Report
    // --------------------------------------------------------

    printf("\n=== Timing (avg over %d iters) ===\n", iters);
    printf("Naive attention (QK^T + softmax + PV): %.4f ms\n", ms_naive);
    printf("Fused attention (single kernel)       : %.4f ms\n", ms_fused);
    printf("Speedup (naive / fused)               : %.2fx\n",
           ms_naive / ms_fused);

    // rough FLOP count (for reference, not super precise)
    double flops_qk = 2.0 * (double)S * S * D; // mul+add
    double flops_softmax = (double)S * S * 4.0; // exp/add/div rough
    double flops_pv = 2.0 * (double)S * S * D;

    double total_flops = flops_qk + flops_softmax + flops_pv;
    double t_naive_s = ms_naive * 1e-3;
    double t_fused_s = ms_fused * 1e-3;

    double tflops_naive = total_flops / t_naive_s / 1e12;
    double tflops_fused = total_flops / t_fused_s / 1e12;

    printf("\nApprox attention TFLOP/s (both counted with same FLOPs)\n");
    printf("Naive : %.3f TFLOP/s\n", tflops_naive);
    printf("Fused : %.3f TFLOP/s\n", tflops_fused);

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_O_naive));
    CUDA_CHECK(cudaFree(d_O_fused));

    return 0;
}
/*
nvcc -O3 -arch=sm_86 attention_fusion_test.cu -o attention_fusion_test.exe

# 기본: S=512, D=64, iters=100
./attention_fusion_test.exe

# ncu 프로파일링 (naive QK^T)
ncu --kernel-name regex:naive_qk_kernel.*     --set full     --launch-skip 5 --launch-count 1     ./attention_fusion_test.exe 512 64 100

# ncu 프로파일링 (fused)
ncu --kernel-name regex:fused_attention_kernel.*     --set full     --launch-skip 5 --launch-count 1     ./attention_fusion_test.exe 512 64 100

*/