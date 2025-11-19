// gemm_perf.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do {                                      \
  cudaError_t _e = (call);                                         \
  if (_e != cudaSuccess) {                                         \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
            __FILE__, __LINE__, cudaGetErrorString(_e));           \
    std::exit(EXIT_FAILURE);                                       \
  }                                                                \
} while (0)
#endif

// -----------------------------------------------------------------------------
// Naive GEMM: C = A(MxK) * B(KxN)  (row-major)
// -----------------------------------------------------------------------------
template<typename T>
__global__ void gemm_naive_kernel(
    int M, int N, int K,
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    T acc = T(0);
    int a_row_offset = row * K;
    int b_col_offset = col;
    for (int k = 0; k < K; ++k) {
        acc += A[a_row_offset + k] * B[k * N + b_col_offset];
    }
    C[row * N + col] = acc;
}

// -----------------------------------------------------------------------------
// Tiled GEMM (shared memory): block = TILE x TILE, thread당 C 1원소
// -----------------------------------------------------------------------------
template<typename T, int TILE>
__global__ void gemm_tiled_kernel(
    int M, int N, int K,
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C)
{
    __shared__ T As[TILE][TILE];
    __shared__ T Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    T acc = T(0);

    int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        int kA = t * TILE + threadIdx.x; // A: (row, kA)
        int kB = t * TILE + threadIdx.y; // B: (kB, col)

        if (row < M && kA < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + kA];
        } else {
            As[threadIdx.y][threadIdx.x] = T(0);
        }

        if (kB < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[kB * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = T(0);
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// -----------------------------------------------------------------------------
// Tiled + Register tiling (1 x TN) : thread당 연속된 TN개 column 담당
// -----------------------------------------------------------------------------
template<typename T, int TILE, int TN>
__global__ void gemm_tiled_regN_kernel(
    int M, int N, int K,
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C)
{
    static_assert(TILE % TN == 0, "TILE must be divisible by TN");

    // block tile: TILE x TILE (M x N)
    // blockDim.x = TILE / TN, blockDim.y = TILE
    __shared__ T As[TILE][TILE]; // [row][k_tile]
    __shared__ T Bs[TILE][TILE]; // [k_tile][col]

    int tx = threadIdx.x;              // 0 .. (TILE/TN - 1)
    int ty = threadIdx.y;              // 0 .. TILE-1

    int block_row = blockIdx.y * TILE;
    int block_col = blockIdx.x * TILE;

    int row  = block_row + ty;
    int col0 = block_col + tx * TN;    // thread가 담당하는 첫 column

    // 레지스터 accumulator: 1 x TN
    T acc[TN];
    #pragma unroll
    for (int j = 0; j < TN; ++j) acc[j] = T(0);

    int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        int k0 = t * TILE;

        // -----------------------------
        // As 로드: (row, k0 .. k0+TILE-1)
        // Bs 로드: (k0 .. k0+TILE-1, col0 .. col0+TN-1)
        // -----------------------------
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int kk   = tx * TN + j;       // 0 .. TILE-1
            int k_gl = k0 + kk;

            // A tile
            if (row < M && k_gl < K) {
                As[ty][kk] = A[row * K + k_gl];
            } else {
                As[ty][kk] = T(0);
            }

            // B tile: ty를 k index로 사용
            int col = col0 + j;
            int kB  = k0 + ty;
            if (kB < K && col < N) {
                Bs[ty][kk] = B[kB * N + col];
            } else {
                Bs[ty][kk] = T(0);
            }
        }

        __syncthreads();

        // -----------------------------
        // K 방향 타일 곱 (TILE 만큼)
        // C(row, col0..col0+TN-1) 업데이트
        // -----------------------------
        #pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            T a = As[ty][kk];
            // Bs[kk][n_idx] 로부터 TN개의 column 사용
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int n_idx = tx * TN + j;
                T b = Bs[kk][n_idx];
                acc[j] += a * b;
            }
        }

        __syncthreads();
    }

    // 결과 쓰기
    if (row < M) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int col = col0 + j;
            if (col < N) {
                C[row * N + col] = acc[j];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Host helpers
// -----------------------------------------------------------------------------
float run_gemm_naive(int M, int N, int K,
                     const float* dA, const float* dB, float* dC,
                     int iters)
{
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    // warmup
    for (int i = 0; i < 3; ++i) {
        gemm_naive_kernel<float><<<grid, block>>>(M, N, K, dA, dB, dC);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        gemm_naive_kernel<float><<<grid, block>>>(M, N, K, dA, dB, dC);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / iters;
}

template<int TILE>
float run_gemm_tiled(int M, int N, int K,
                     const float* dA, const float* dB, float* dC,
                     int iters)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE,
              (M + TILE - 1) / TILE);

    // warmup
    for (int i = 0; i < 3; ++i) {
        gemm_tiled_kernel<float, TILE><<<grid, block>>>(M, N, K, dA, dB, dC);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        gemm_tiled_kernel<float, TILE><<<grid, block>>>(M, N, K, dA, dB, dC);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / iters;
}

// 1 x TN register tiling 버전
template<int TILE, int TN>
float run_gemm_tiled_regN(int M, int N, int K,
                          const float* dA, const float* dB, float* dC,
                          int iters)
{
    static_assert(TILE % TN == 0, "TILE must be divisible by TN");

    dim3 block(TILE / TN, TILE); // (TILE/TN, TILE)
    dim3 grid((N + TILE - 1) / TILE,
              (M + TILE - 1) / TILE);

    // warmup
    for (int i = 0; i < 3; ++i) {
        gemm_tiled_regN_kernel<float, TILE, TN><<<grid, block>>>(
            M, N, K, dA, dB, dC);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        gemm_tiled_regN_kernel<float, TILE, TN><<<grid, block>>>(
            M, N, K, dA, dB, dC);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / iters;
}

void check_max_diff(const float* h_ref, const float* h_out, int M, int N)
{
    double max_diff = 0.0;
    long long total = (long long)M * N;
    for (long long i = 0; i < total; ++i) {
        double d = std::fabs((double)h_ref[i] - (double)h_out[i]);
        if (d > max_diff) max_diff = d;
    }
    printf("  max |diff| = %e\n", max_diff);
}

// -----------------------------------------------------------------------------
// 한 사이즈(M,N,K)에 대해 전체 실험 수행
// -----------------------------------------------------------------------------
template<int TILE>
void run_one_case(int M, int N, int K, int iters)
{
    constexpr int TN = 4; // thread당 column 개수 (1 x TN 레지스터 타일)

    printf("============================================================\n");
    printf("Case: M = %d, N = %d, K = %d, TILE = %d, TN = %d, iters = %d\n",
           M, N, K, TILE, TN, iters);

    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);

    float* hA       = (float*)malloc(sizeA);
    float* hB       = (float*)malloc(sizeB);
    float* hC_naive = (float*)malloc(sizeC);
    float* hC_tiled = (float*)malloc(sizeC);
    float* hC_reg   = (float*)malloc(sizeC);

    // 호스트 초기화
    for (int i = 0; i < M * K; ++i) {
        hA[i] = (float)((i % 13) - 6);
    }
    for (int i = 0; i < K * N; ++i) {
        hB[i] = (float)((i % 7) - 3);
    }

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, sizeA));
    CHECK_CUDA(cudaMalloc(&dB, sizeB));
    CHECK_CUDA(cudaMalloc(&dC, sizeC));

    CHECK_CUDA(cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice));

    // Naive
    float ms_naive = run_gemm_naive(M, N, K, dA, dB, dC, iters);
    CHECK_CUDA(cudaMemcpy(hC_naive, dC, sizeC, cudaMemcpyDeviceToHost));

    // Tiled (shared only)
    float ms_tiled = run_gemm_tiled<TILE>(M, N, K, dA, dB, dC, iters);
    CHECK_CUDA(cudaMemcpy(hC_tiled, dC, sizeC, cudaMemcpyDeviceToHost));

    // Tiled + Register (1 x TN)
    float ms_reg = run_gemm_tiled_regN<TILE, TN>(M, N, K, dA, dB, dC, iters);
    CHECK_CUDA(cudaMemcpy(hC_reg, dC, sizeC, cudaMemcpyDeviceToHost));

    // FLOPs, GFLOPs
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops_naive = flops / (ms_naive * 1e-3) / 1e9;
    double gflops_tiled = flops / (ms_tiled * 1e-3) / 1e9;
    double gflops_reg   = flops / (ms_reg   * 1e-3) / 1e9;

    printf("  [naive]     avg time: %.4f ms, %.2f GFLOP/s\n", ms_naive, gflops_naive);
    printf("  [tiled]     avg time: %.4f ms, %.2f GFLOP/s\n", ms_tiled, gflops_tiled);
    printf("  [tiled+reg] avg time: %.4f ms, %.2f GFLOP/s\n", ms_reg,   gflops_reg);
    printf("  tiled / naive     = %.4fx (time)\n", ms_tiled / ms_naive);
    printf("  tiled+reg / naive = %.4fx (time)\n", ms_reg   / ms_naive);
    printf("  tiled+reg / tiled = %.4fx (time)\n", ms_reg   / ms_tiled);

    printf("  diff naive vs tiled:\n");
    check_max_diff(hC_naive, hC_tiled, M, N);
    printf("  diff naive vs tiled+reg:\n");
    check_max_diff(hC_naive, hC_reg,   M, N);

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    free(hA);
    free(hB);
    free(hC_naive);
    free(hC_tiled);
    free(hC_reg);
}

int main()
{
    const int TILE = 32;  // TILE % TN == 0 이어야 함 (TN=4)

    struct CaseCfg {
        int M, N, K, iters;
    };

    CaseCfg cases[] = {
        //{512,  512,  512,  100},
        {1024, 1024, 1024, 50},
        {2048, 2048, 2048, 20},
        {4096, 4096, 4096, 5},  // GPU 메모리 여유되면 사용
    };

    int num_cases = sizeof(cases) / sizeof(cases[0]);

    for (int i = 0; i < num_cases; ++i) {
        run_one_case<TILE>(cases[i].M, cases[i].N, cases[i].K, cases[i].iters);
    }

    return 0;
}
