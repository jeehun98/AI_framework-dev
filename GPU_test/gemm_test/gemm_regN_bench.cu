// gemm_regN_bench.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ================================================================
// 유틸
// ================================================================
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

// 컴파일 시 바꾸고 싶으면 -DGEMM_TILE=64 -DGEMM_TN=8 이런 식으로
#ifndef GEMM_TILE
#define GEMM_TILE 32
#endif

#ifndef GEMM_TN
#define GEMM_TN 4
#endif

// ================================================================
// 3) Tiled + 1 x TN register tiling GEMM
// ================================================================
template<typename T, int TILE, int TN>
__global__ void gemm_tiled_regN_kernel(
    int M, int N, int K,
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C)
{
    static_assert(TILE % TN == 0, "TILE must be divisible by TN");

    __shared__ T As[TILE][TILE+1]; // [row][k_tile]
    __shared__ T Bs[TILE][TILE+1]; // [k_tile][col]

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

        // As, Bs 로드
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int kk   = tx * TN + j;   // 0 .. TILE-1
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

        // K 타일 내 곱셈
        #pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            T a = As[ty][kk];
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

// ================================================================
// CPU ref GEMM (검증용)
// ================================================================
void gemm_cpu_ref(
    int M, int N, int K,
    const float* A,
    const float* B,
    float* C)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += (double)A[i * K + k] * (double)B[k * N + j];
            }
            C[i * N + j] = (float)acc;
        }
    }
}

double max_abs_diff(const float* ref, const float* out, int M, int N)
{
    double max_diff = 0.0;
    long long total = (long long)M * N;
    for (long long i = 0; i < total; ++i) {
        double d = std::fabs((double)ref[i] - (double)out[i]);
        if (d > max_diff) max_diff = d;
    }
    return max_diff;
}

// ================================================================
// Host-side runner: N회 반복 성능 측정
// ================================================================
template<int TILE, int TN>
float run_gemm_tiled_regN(
    int M, int N, int K,
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

// ================================================================
// main
//  usage:
//    ./gemm_regN_bench [M N K [iters_bench [iters_profile]]]
//  예:
//    ./gemm_regN_bench             // 1024 1024 1024, bench=50, profile=1
//    ./gemm_regN_bench 2048 2048 2048 100 0
//    ncu ... ./gemm_regN_bench 1024 1024 1024 10 1
// ================================================================
int main(int argc, char** argv)
{
    const int TILE = GEMM_TILE;
    const int TN   = GEMM_TN;

    int M = 1024;
    int N = 1024;
    int K = 1024;
    int iters_bench   = 50;  // 평균 시간 / GFLOP/s 측정용
    int iters_profile = 1;   // Nsight 관찰용 (1회 또는 소수 회)

    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    if (argc >= 5) {
        iters_bench = std::atoi(argv[4]);
    }
    if (argc >= 6) {
        iters_profile = std::atoi(argv[5]);
    }

    printf("==== GEMM regN kernel benchmark ====\n");
    printf("  M=%d, N=%d, K=%d\n", M, N, K);
    printf("  TILE=%d, TN=%d\n", TILE, TN);
    printf("  iters_bench   = %d\n", iters_bench);
    printf("  iters_profile = %d\n", iters_profile);

    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);

    float* hA      = (float*)malloc(sizeA);
    float* hB      = (float*)malloc(sizeB);
    float* hC_ref  = (float*)malloc(sizeC);
    float* hC_out  = (float*)malloc(sizeC);

    if (!hA || !hB || !hC_ref || !hC_out) {
        fprintf(stderr, "Host malloc failed\n");
        return EXIT_FAILURE;
    }

    // 호스트 초기화 (repeatable pattern)
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

    // ------------------------------------------------------------
    // 1) 성능 측정 (N회 반복)
    // ------------------------------------------------------------
    float ms_avg = run_gemm_tiled_regN<TILE, TN>(
        M, N, K, dA, dB, dC, iters_bench);

    CHECK_CUDA(cudaMemcpy(hC_out, dC, sizeC, cudaMemcpyDeviceToHost));

    double flops = 2.0 * (double)M * (double)N * (double)K;
    double sec   = ms_avg * 1e-3;
    double gflops = flops / sec / 1e9;

    printf("\n[Benchmark]\n");
    printf("  avg time: %.4f ms\n", ms_avg);
    printf("  perf    : %.2f GFLOP/s\n", gflops);

    // ------------------------------------------------------------
    // 2) CPU ref + diff 체크 (한 번만)
    //    (원하면 주석 처리해서 profiling에 영향 없게 해도 됨)
    // ------------------------------------------------------------
    printf("\n[Check] CPU reference vs GPU\n");
    gemm_cpu_ref(M, N, K, hA, hB, hC_ref);
    double max_diff = max_abs_diff(hC_ref, hC_out, M, N);
    printf("  max |diff| = %e\n", max_diff);

    // ------------------------------------------------------------
    // 3) Nsight용 프로파일 실행
    //    - iters_profile 번 돌리고 동기화만 해줌
    //    - 여기서 Nsight Compute / Systems 붙여서 타임라인/metrics 확인
    // ------------------------------------------------------------
    if (iters_profile > 0) {
        printf("\n[Profile] running kernel %d time(s) for Nsight...\n",
               iters_profile);

        dim3 block(TILE / TN, TILE);
        dim3 grid((N + TILE - 1) / TILE,
                  (M + TILE - 1) / TILE);

        for (int i = 0; i < iters_profile; ++i) {
            gemm_tiled_regN_kernel<float, TILE, TN><<<grid, block>>>(
                M, N, K, dA, dB, dC);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        printf("  [Profile] done.\n");
    }

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    free(hA);
    free(hB);
    free(hC_ref);
    free(hC_out);

    return 0;
}

/*
빌드 예시 (sm_86 기준):

  nvcc -O3 -arch=sm_86 gemm_regN_bench.cu -o gemm_regN_bench

실행 예시:

  // 기본 (1024^3, bench=50, profile=1)
  ./gemm_regN_bench

  // 2048^3, bench=100번, profile 안 돌림
  ./gemm_regN_bench 2048 2048 2048 100 0

  // Nsight Compute로 1024^3, bench 10번 + profile 1번
  ncu --set full --target-processes all --export output.ncu-rep ./gemm_regN_bench 1024 1024 1024 10 1

*/
