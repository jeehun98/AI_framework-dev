// gemm_register_reuse_test.cu
// Test 2. Register reuse via multi-accumulators (TN sweep)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                            \
                    __FILE__, __LINE__, cudaGetErrorString(err__));              \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

// Simple host GEMM for reference: C = A * B
void host_gemm_ref(const float* A, const float* B, float* C,
                   int M, int N, int K)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

// TN-way multi-accumulator GEMM without SMEM.
// Each thread computes TN outputs along N dimension,
// and reuses the same A(row,k) value across TN FMAs.
template<int TN>
__global__ void gemm_regreuse_TN_kernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int M, int N, int K)
{
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16 * TN;

    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    int local_row = threadIdx.y;          // 0..15
    int local_col = threadIdx.x;          // 0..15

    int row = block_row * TILE_M + local_row;
    int col_base = block_col * TILE_N + local_col * TN; // TN outputs per thread

    if (row >= M) return;

    // TN accumulators per thread
    float acc[TN];
    #pragma unroll
    for (int t = 0; t < TN; ++t) {
        acc[t] = 0.f;
    }

    for (int k = 0; k < K; ++k) {
        // One load of A(row, k), reused across TN FMAs
        float a = A[row * K + k];

        #pragma unroll
        for (int t = 0; t < TN; ++t) {
            int col = col_base + t;
            if (col < N) {
                float b = B[k * N + col];
                acc[t] = fmaf(a, b, acc[t]);
            }
        }
    }

    // Write back results
    #pragma unroll
    for (int t = 0; t < TN; ++t) {
        int col = col_base + t;
        if (col < N) {
            C[row * N + col] = acc[t];
        }
    }
}

float max_abs_diff(const float* a, const float* b, int n)
{
    float m = 0.f;
    for (int i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main()
{
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    const size_t bytes_A = size_t(M) * K * sizeof(float);
    const size_t bytes_B = size_t(K) * N * sizeof(float);
    const size_t bytes_C = size_t(M) * N * sizeof(float);

    printf("=== Register Reuse Test 2: Multi-accumulator (TN=1 vs TN=4) ===\n");
    printf("GEMM: C[%d x %d] = A[%d x %d] * B[%d x %d]\n", M, N, M, K, K, N);
    printf("Each thread computes TN outputs along N and reuses A(row,k) across TN FMAs.\n\n");

    // Allocate host memory
    float* h_A    = (float*)std::malloc(bytes_A);
    float* h_B    = (float*)std::malloc(bytes_B);
    float* h_Cref = (float*)std::malloc(bytes_C);
    float* h_C    = (float*)std::malloc(bytes_C);

    if (!h_A || !h_B || !h_Cref || !h_C) {
        fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize A, B
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = (float)((i % 13) - 6) * 0.1f; // some pattern
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = (float)((i % 17) - 8) * 0.05f;
    }

    printf("Computing host reference...\n");
    host_gemm_ref(h_A, h_B, h_Cref, M, N, K);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C, bytes_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, bytes_C));

    dim3 block(16, 16);

    // ===== TN = 1 (baseline, 최소 레지스터 재사용) =====
    {
        constexpr int TN = 1;
        constexpr int TILE_M = 16;
        constexpr int TILE_N = 16 * TN;

        dim3 grid(N / TILE_N, M / TILE_M);

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        // warm-up
        gemm_regreuse_TN_kernel<TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemset(d_C, 0, bytes_C));

        CHECK_CUDA(cudaEventRecord(start));
        gemm_regreuse_TN_kernel<TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
        float max_diff = max_abs_diff(h_C, h_Cref, M * N);

        printf("[TN=1]  grid=(%d,%d), block=(%d,%d)\n",
               grid.x, grid.y, block.x, block.y);
        printf("[TN=1]  time = %.3f ms, max diff = %.6e\n\n", ms, max_diff);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    // ===== TN = 4 (강한 레지스터 재사용: A(row,k) → 4개의 C 출력에 사용) =====
    {
        constexpr int TN = 4;
        constexpr int TILE_M = 16;
        constexpr int TILE_N = 16 * TN;

        dim3 grid(N / TILE_N, M / TILE_M);

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaMemset(d_C, 0, bytes_C));

        // warm-up
        gemm_regreuse_TN_kernel<TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemset(d_C, 0, bytes_C));

        CHECK_CUDA(cudaEventRecord(start));
        gemm_regreuse_TN_kernel<TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
        float max_diff = max_abs_diff(h_C, h_Cref, M * N);

        printf("[TN=4]  grid=(%d,%d), block=(%d,%d)\n",
               grid.x, grid.y, block.x, block.y);
        printf("[TN=4]  time = %.3f ms, max diff = %.6e\n\n", ms, max_diff);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    std::free(h_A);
    std::free(h_B);
    std::free(h_Cref);
    std::free(h_C);

    return 0;
}

/*
빌드 예시 (Windows, nvcc):

  nvcc -O3 -arch=sm_86 gemm_register_reuse_test.cu -o gemm_register_reuse_test.exe

ncu 예시:


ncu --kernel-name regex:gemm_regreuse_TN_kernel.* --metrics smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ./gemm_register_reuse_test.exe

여기서 TN=1 vs TN=4 의 kernel time / FMA pipe 활성 비율 / gld_throughput 을 비교하면
"한 번 load된 A(row,k)를 여러 accumulator에 재사용"하는 구조의 효과가 바로 드러난다.
*/
