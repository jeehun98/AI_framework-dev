#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err__));            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// ----------------------------
// Config (attention-like)
// ----------------------------
constexpr int B  = 4;    // batch
constexpr int H  = 8;    // heads
constexpr int S  = 1024; // sequence length
constexpr int D  = 64;   // head dimension

// 각 (b,h)에 대해 S * D 크기의 Q,K,V를 갖는 형태를 가정
// Q, K, V: [B*H, S, D]
// QKV_packed: [B*H, S, 3, D] -> [B*H * S, 3*D]

// ----------------------------
// Device kernels
// ----------------------------

// 분리형 Q/K/V 레이아웃
// 각 블록: (b,h) 한 쌍 처리
// 단순히 Q/K/V를 모두 읽어서 합산(accumulate)만 하는 패턴
__global__ void attention_qkv_separate_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ out,
    int B, int H, int S, int D)
{
    int bh = blockIdx.x;          // 0..B*H-1
    int tid = threadIdx.x;
    int BH = B * H;

    if (bh >= BH) return;

    float acc = 0.f;

    // Q pass (전체 시퀀스 스트림)
    for (int s = 0; s < S; ++s) {
        int row_base = (bh * S + s) * D;
        for (int d = tid; d < D; d += blockDim.x) {
            float q = Q[row_base + d];
            acc += q * 0.1f;
        }
    }

    // K pass
    for (int s = 0; s < S; ++s) {
        int row_base = (bh * S + s) * D;
        for (int d = tid; d < D; d += blockDim.x) {
            float k = K[row_base + d];
            acc += k * 0.2f;
        }
    }

    // V pass
    for (int s = 0; s < S; ++s) {
        int row_base = (bh * S + s) * D;
        for (int d = tid; d < D; d += blockDim.x) {
            float v = V[row_base + d];
            acc += v * 0.3f;
        }
    }

    // 단순히 각 thread별 결과를 out에 기록 (검증용/최적화 방지)
    int out_stride = blockDim.x;
    int out_idx    = bh * out_stride + tid;
    out[out_idx]   = acc;
}

// QKV packed 레이아웃
// QKV_packed: [B*H, S, 3, D] 를 일렬로 둔 형태
// 한 pass 안에서 Q/K/V를 같이 읽게 해서 layout 차이에 따른 locality를 실험
__global__ void attention_qkv_packed_kernel(
    const float* __restrict__ QKV_packed,
    float* __restrict__ out,
    int B, int H, int S, int D)
{
    int bh = blockIdx.x; // 0..B*H-1
    int tid = threadIdx.x;
    int BH  = B * H;

    if (bh >= BH) return;

    float acc = 0.f;
    int row_stride = 3 * D;  // [3, D] 부분

    for (int s = 0; s < S; ++s) {
        int base = (bh * S + s) * row_stride;
        for (int d = tid; d < D; d += blockDim.x) {
            float q = QKV_packed[base + d];         // Q
            float k = QKV_packed[base + D + d];     // K
            float v = QKV_packed[base + 2*D + d];   // V
            acc += q * 0.1f + k * 0.2f + v * 0.3f;
        }
    }

    int out_stride = blockDim.x;
    int out_idx    = bh * out_stride + tid;
    out[out_idx]   = acc;
}

// ----------------------------
// Host side helpers
// ----------------------------

void init_random(float* ptr, size_t n, float scale = 1.0f) {
    for (size_t i = 0; i < n; ++i) {
        ptr[i] = scale * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f);
    }
}

int main() {
    std::srand(42);

    const int BH = B * H;
    const size_t elems_per_tensor = static_cast<size_t>(BH) * S * D;
    const size_t bytes_per_tensor = elems_per_tensor * sizeof(float);

    printf("=== L2 Locality Test 2: Attention QKV layout (separate vs packed) ===\n");
    printf("B=%d, H=%d, S=%d, D=%d\n", B, H, S, D);
    printf("Total Q/K/V elements each: %zu (%.2f MB)\n",
           elems_per_tensor,
           bytes_per_tensor / (1024.0 * 1024.0));

    // Host memory
    float* h_Q = (float*)std::malloc(bytes_per_tensor);
    float* h_K = (float*)std::malloc(bytes_per_tensor);
    float* h_V = (float*)std::malloc(bytes_per_tensor);
    float* h_QKV_packed = (float*)std::malloc(bytes_per_tensor * 3); // 3 * D

    if (!h_Q || !h_K || !h_V || !h_QKV_packed) {
        fprintf(stderr, "Host malloc failed\n");
        return EXIT_FAILURE;
    }

    init_random(h_Q, elems_per_tensor, 1.0f);
    init_random(h_K, elems_per_tensor, 1.0f);
    init_random(h_V, elems_per_tensor, 1.0f);

    // QKV packed 만들기: [BH, S, 3, D]
    {
        for (int bh = 0; bh < BH; ++bh) {
            for (int s = 0; s < S; ++s) {
                size_t base_separate = (static_cast<size_t>(bh) * S + s) * D;
                size_t base_packed   = (static_cast<size_t>(bh) * S + s) * (3 * D);

                for (int d = 0; d < D; ++d) {
                    h_QKV_packed[base_packed + d]       = h_Q[base_separate + d]; // Q
                    h_QKV_packed[base_packed + D + d]   = h_K[base_separate + d]; // K
                    h_QKV_packed[base_packed + 2*D + d] = h_V[base_separate + d]; // V
                }
            }
        }
    }

    // Device memory
    float *d_Q, *d_K, *d_V, *d_QKV_packed, *d_out_sep, *d_out_packed;
    CUDA_CHECK(cudaMalloc(&d_Q, bytes_per_tensor));
    CUDA_CHECK(cudaMalloc(&d_K, bytes_per_tensor));
    CUDA_CHECK(cudaMalloc(&d_V, bytes_per_tensor));
    CUDA_CHECK(cudaMalloc(&d_QKV_packed, bytes_per_tensor * 3));

    // out: (B*H blocks * blockDim.x threads) 결과 저장
    int block_dim = 128;
    size_t out_elems = static_cast<size_t>(BH) * block_dim;
    size_t out_bytes = out_elems * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_out_sep, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_packed, out_bytes));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, bytes_per_tensor, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, bytes_per_tensor, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, bytes_per_tensor, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_QKV_packed, h_QKV_packed, bytes_per_tensor * 3,
                          cudaMemcpyHostToDevice));

    dim3 grid(BH);
    dim3 block(block_dim);

    // Warm-up
    attention_qkv_separate_kernel<<<grid, block>>>(
        d_Q, d_K, d_V, d_out_sep, B, H, S, D);
    attention_qkv_packed_kernel<<<grid, block>>>(
        d_QKV_packed, d_out_packed, B, H, S, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timing helper
    auto run_and_time = [](auto kernel,
                           dim3 grid, dim3 block,
                           void** args,
                           int repeat, const char* label) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // warm-up
        for (int i = 0; i < 3; ++i) {
            CUDA_CHECK(cudaLaunchKernel((const void*)kernel, grid, block,
                                        args, 0, nullptr));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < repeat; ++i) {
            CUDA_CHECK(cudaLaunchKernel((const void*)kernel, grid, block,
                                        args, 0, nullptr));
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= repeat;

        printf("[%s] avg time = %.3f ms\n", label, ms);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    };

    // Separate Q/K/V timing
    {
        void* args[] = {
            (void*)&d_Q,
            (void*)&d_K,
            (void*)&d_V,
            (void*)&d_out_sep,
            (void*)&B,
            (void*)&H,
            (void*)&S,
            (void*)&D
        };
        run_and_time((void*)attention_qkv_separate_kernel,
                     grid, block, args, /*repeat=*/10,
                     "QKV_separate");
    }

    // Packed QKV timing
    {
        void* args[] = {
            (void*)&d_QKV_packed,
            (void*)&d_out_packed,
            (void*)&B,
            (void*)&H,
            (void*)&S,
            (void*)&D
        };
        run_and_time((void*)attention_qkv_packed_kernel,
                     grid, block, args, /*repeat=*/10,
                     "QKV_packed");
    }

    // 간단히 결과 일부만 확인해서 최적화가 완전히 사라지지 않도록
    float* h_out = (float*)std::malloc(out_bytes);
    CUDA_CHECK(cudaMemcpy(h_out, d_out_sep, out_bytes, cudaMemcpyDeviceToHost));
    printf("sample out_separate[0] = %f\n", h_out[0]);
    CUDA_CHECK(cudaMemcpy(h_out, d_out_packed, out_bytes, cudaMemcpyDeviceToHost));
    printf("sample out_packed[0]   = %f\n", h_out[0]);

    std::free(h_Q);
    std::free(h_K);
    std::free(h_V);
    std::free(h_QKV_packed);
    std::free(h_out);

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_QKV_packed));
    CUDA_CHECK(cudaFree(d_out_sep));
    CUDA_CHECK(cudaFree(d_out_packed));

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
/*

nvcc -O3 attention_qkv_l2_layout_test.cu -o attention_qkv_l2_layout_test.exe

# 분리형 Q/K/V
ncu --kernel-name regex:attention_qkv_separate_kernel.*     --metrics dram__bytes_read.sum,lts__t_sectors_hit_rate.pct     ./attention_qkv_l2_layout_test.exe

# QKV packed
ncu --kernel-name regex:attention_qkv_packed_kernel.*     --metrics dram__bytes_read.sum,lts__t_sectors_hit_rate.pct     ./attention_qkv_l2_layout_test.exe
*/