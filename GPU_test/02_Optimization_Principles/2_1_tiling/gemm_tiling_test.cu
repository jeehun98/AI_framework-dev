#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error %s:%d: %s\n",                           \
                    __FILE__, __LINE__, cudaGetErrorString(err__));             \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)
#endif

// =======================
// 매트릭스 사이즈 설정
// =======================
// C[M x N] = A[M x K] * B[K x N]
constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;

// shared tiling block size
constexpr int BLOCK_SIZE = 32;

// =======================
// naive GEMM kernel
// =======================
// thread당 C(row, col) 하나 계산, 모든 K 루프를 global memory에서 직접 로드
__global__ void gemm_naive_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // [0, M)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // [0, N)

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    // A[row, k], B[k, col]
    for (int k = 0; k < K; ++k) {
        float a = A[row * K + k];
        float b = B[k * N + col];
        acc += a * b;
    }
    C[row * N + col] = acc;
}

// =======================
// shared-memory tiling GEMM kernel
// =======================
// tile 크기: BLOCK_SIZE x BLOCK_SIZE
// 각 block이 C의 BLOCK_SIZE x BLOCK_SIZE tile 담당
__global__ void gemm_shared_tiling_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // block 기준 row / col
    int row = blockIdx.y * blockDim.y + threadIdx.y; // [0, M)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // [0, N)

    // shared memory tile
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    float acc = 0.0f;

    // K dimension을 BLOCK_SIZE 단위로 쪼개서 loop
    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int kA = t * BLOCK_SIZE + threadIdx.x; // A에서 column index (K 방향)
        int kB = t * BLOCK_SIZE + threadIdx.y; // B에서 row index (K 방향)

        // A tile 로드: A[row, kA]
        if (row < M && kA < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + kA];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // B tile 로드: B[kB, col]
        if (kB < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[kB * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // tile 내에서 K 방향 축소
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// =======================
// 유틸: 호스트 초기화 & 검증
// =======================
void init_matrix(float* mat, int rows, int cols, float scale = 1.0f) {
    for (int i = 0; i < rows * cols; ++i) {
        // 간단한 deterministic 패턴 (Nsight에서 패턴 안 중요)
        mat[i] = scale * ((i % 13) - 6);  // -6 ~ +6
    }
}

float max_abs_diff(const float* ref, const float* test, int n) {
    float max_diff = 0.0f;
    for (int i = 0; i < n; ++i) {
        float d = std::fabs(ref[i] - test[i]);
        if (d > max_diff) max_diff = d;
    }
    return max_diff;
}

// 단순 host-side GEMM (검증용, 느려도 상관 없음, 한 번만 실행)
void gemm_host_ref(const float* A, const float* B, float* C,
                   int M, int N, int K) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = acc;
        }
    }
}

// =======================
// main: naive vs shared_tiling 비교
// =======================
int main() {
    printf("GEMM config: C[%d x %d] = A[%d x %d] * B[%d x %d]\n",
           M, N, M, K, K, N);
    printf("BLOCK_SIZE = %d\n", BLOCK_SIZE);

    size_t bytesA = sizeof(float) * M * K;
    size_t bytesB = sizeof(float) * K * N;
    size_t bytesC = sizeof(float) * M * N;

    float *hA = (float*)malloc(bytesA);
    float *hB = (float*)malloc(bytesB);
    float *hC_naive = (float*)malloc(bytesC);
    float *hC_tiled = (float*)malloc(bytesC);
    float *hC_ref   = (float*)malloc(bytesC);

    init_matrix(hA, M, K, 0.01f);
    init_matrix(hB, K, N, 0.02f);

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));

    CHECK_CUDA(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // -----------------------
    // host reference (검증용)
    // -----------------------
    printf("Computing host reference...\n");
    gemm_host_ref(hA, hB, hC_ref, M, N, K);

    // -----------------------
    // naive kernel
    // -----------------------
    {
        printf("\n[naive] warm-up + timing\n");
        // warm-up
        gemm_naive_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        gemm_naive_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        printf("[naive] kernel time: %.3f ms\n", ms);

        CHECK_CUDA(cudaMemcpy(hC_naive, dC, bytesC, cudaMemcpyDeviceToHost));
        float diff = max_abs_diff(hC_ref, hC_naive, M * N);
        printf("[naive] max |C_ref - C_naive| = %e\n", diff);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    // -----------------------
    // shared tiling kernel
    // -----------------------
    {
        printf("\n[shared_tiling] warm-up + timing\n");
        // warm-up
        gemm_shared_tiling_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        gemm_shared_tiling_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        printf("[shared_tiling] kernel time: %.3f ms\n", ms);

        CHECK_CUDA(cudaMemcpy(hC_tiled, dC, bytesC, cudaMemcpyDeviceToHost));
        float diff = max_abs_diff(hC_ref, hC_tiled, M * N);
        printf("[shared_tiling] max |C_ref - C_tiled| = %e\n", diff);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    // =======================
    // cleanup
    // =======================
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    free(hA);
    free(hB);
    free(hC_naive);
    free(hC_tiled);
    free(hC_ref);

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}

// nvcc -O3 -arch=sm_86 -o gemm_tiling_test.exe gemm_tiling_test.cu
