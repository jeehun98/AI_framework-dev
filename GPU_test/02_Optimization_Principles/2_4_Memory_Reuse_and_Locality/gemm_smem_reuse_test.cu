// gemm_smem_reuse_test.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>

#define CUDA_CHECK(ans)                                                     \
    {                                                                       \
        cudaError_t err = (ans);                                            \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;

constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 16;

// =======================
// Naive GEMM (No Reuse)
//   - A/B를 매번 global에서 직접 읽음
// =======================
__global__ void gemm_noreuse_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // [0, M)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // [0, N)

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    // 완전 naive: K loop에서 매번 global load
    for (int k = 0; k < K; ++k) {
        float a = A[row * K + k];      // row-major
        float b = B[k * N + col];      // row-major
        acc += a * b;
    }
    C[row * N + col] = acc;
}

// =======================
// Shared Tiling GEMM (SMEM Reuse)
//   - A/B tile을 shared에 올려 K-loop 동안 재사용
// =======================
__global__ void gemm_smem_reuse_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // [0, M)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // [0, N)

    if (row >= M || col >= N) return;

    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    float acc = 0.0f;

    // K dimension을 TILE_K 단위로 순회
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        int ty = threadIdx.y;
        int tx = threadIdx.x;

        // A tile 로드: [row, k0 + tx]
        if (row < M && (k0 + tx) < K) {
            As[ty][tx] = A[row * K + (k0 + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }

        // B tile 로드: [k0 + ty, col]
        if ((k0 + ty) < K && col < N) {
            Bs[ty][tx] = B[(k0 + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // shared tile 재사용: K tile 내부 loop
        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            acc += As[ty][kk] * Bs[kk][tx];
        }

        __syncthreads();
    }

    C[row * N + col] = acc;
}

// =======================
// Host reference GEMM (FP32)
// =======================
void host_gemm_ref(const float* A, const float* B, float* C,
                   int M, int N, int K)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

// =======================
// Max diff 계산
// =======================
float max_abs_diff(const float* a, const float* b, int n)
{
    float m = 0.0f;
    for (int i = 0; i < n; ++i) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main()
{
    printf("=== SMEM Reuse Test 1: No-reuse vs SMEM reuse GEMM ===\n");
    printf("GEMM: C[%d x %d] = A[%d x %d] * B[%d x %d]\n",
           M, N, M, K, K, N);
    printf("Block tile: TILE_M=%d, TILE_N=%d, TILE_K=%d\n",
           TILE_M, TILE_N, TILE_K);

    size_t bytes_A = size_t(M) * K * sizeof(float);
    size_t bytes_B = size_t(K) * N * sizeof(float);
    size_t bytes_C = size_t(M) * N * sizeof(float);

    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C_ref = (float*)malloc(bytes_C);
    float *h_C_noreuse = (float*)malloc(bytes_C);
    float *h_C_smem    = (float*)malloc(bytes_C);

    if (!h_A || !h_B || !h_C_ref || !h_C_noreuse || !h_C_smem) {
        fprintf(stderr, "Host malloc failed\n");
        return EXIT_FAILURE;
    }

    // 간단한 초기화 (deterministic)
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = (float)((i % 13) - 6); // -6 ~ +6 범위
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = (float)((i % 9) - 4);  // -4 ~ +4 범위
    }

    printf("Computing host reference...\n");
    host_gemm_ref(h_A, h_B, h_C_ref, M, N, K);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ======================
    // 1) Naive no-reuse GEMM
    // ======================
    printf("\n[Naive no-reuse GEMM]\n");
    // warm-up
    gemm_noreuse_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    gemm_noreuse_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_noreuse = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_noreuse, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C_noreuse, d_C, bytes_C, cudaMemcpyDeviceToHost));
    float max_diff_noreuse = max_abs_diff(h_C_noreuse, h_C_ref, M * N);

    printf("  time = %.3f ms, max diff vs ref = %e\n",
           ms_noreuse, max_diff_noreuse);

    // ===========================
    // 2) SMEM tiling + reuse GEMM
    // ===========================
    printf("\n[Shared tiling SMEM reuse GEMM]\n");
    // warm-up
    gemm_smem_reuse_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    gemm_smem_reuse_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_smem = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_smem, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C_smem, d_C, bytes_C, cudaMemcpyDeviceToHost));
    float max_diff_smem = max_abs_diff(h_C_smem, h_C_ref, M * N);

    printf("  time = %.3f ms, max diff vs ref = %e\n",
           ms_smem, max_diff_smem);

    // 간단한 요약
    printf("\n=== Summary ===\n");
    printf("Naive(no-reuse):  %.3f ms\n", ms_noreuse);
    printf("SMEM reuse:       %.3f ms\n", ms_smem);
    if (ms_smem > 0.0f) {
        printf("Speedup (no-reuse / SMEM) = %.2fx\n", ms_noreuse / ms_smem);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C_ref);
    free(h_C_noreuse);
    free(h_C_smem);

    return 0;
}
/*
nvcc -O3 -arch=sm_86 gemm_smem_reuse_test.cu -o gemm_smem_reuse_test.exe

# naive (no-reuse) GEMM
ncu --kernel-name regex:gemm_noreuse_kernel.* --metrics dram__bytes_read.sum,lts__t_sectors_pipe_lsu_mem_global_op_ld.sum ./gemm_smem_reuse_test.exe

# shared tiling + reuse GEMM
ncu --kernel-name regex:gemm_smem_reuse_kernel.* --metrics dram__bytes_read.sum,lts__t_sectors_pipe_lsu_mem_global_op_ld.sum ./gemm_smem_reuse_test.exe


*/