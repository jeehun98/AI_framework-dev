#include <cstdio>
#include <cstdlib>
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

// =======================================
// GEMM 설정
// =======================================

constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;

constexpr int BM = 16;
constexpr int BK = 16;

// BlockDim = (16,16), BN = 16 * TN
constexpr int THREADS_X = 16;
constexpr int THREADS_Y = 16;

// =======================================
// host reference GEMM (row-major)
// =======================================

void host_gemm_ref(const float* A, const float* B, float* C,
                   int M, int N, int K)
{
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = acc;
        }
    }
}

// 최대 절대 오차
float max_abs_diff(const float* a, const float* b, int size)
{
    float max_diff = 0.0f;
    for (int i = 0; i < size; ++i) {
        float d = fabsf(a[i] - b[i]);
        if (d > max_diff) max_diff = d;
    }
    return max_diff;
}

// =======================================
// Multi-Accumulator GEMM kernel
// =======================================
//
// 각 block:
//   - 행 방향: BM = 16
//   - 열 방향: BN = 16 * TN
//   - K 방향: BK = 16
//
// 각 thread:
//   - 하나의 행(row)을 담당
//   - 연속된 열 TN개를 담당 (C[row, col0 .. col0+TN-1])
//
// shared memory:
//   - As[BM][BK]
//   - Bs[BK][BN]
//
// =======================================

template<int TN>
__global__ void gemm_multi_acc_TN_kernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int M, int N, int K)
{
    constexpr int BN = THREADS_X * TN;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int tx = threadIdx.x; // 0..15
    const int ty = threadIdx.y; // 0..15

    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    const int row = block_row * BM + ty;
    const int col0 = block_col * BN + tx * TN;

    float acc[TN];
#pragma unroll
    for (int t = 0; t < TN; ++t) {
        acc[t] = 0.0f;
    }

    // K dimension을 BK씩 나누어 타일 루프
    const int num_tiles_k = (K + BK - 1) / BK;

    for (int tile_k = 0; tile_k < num_tiles_k; ++tile_k) {
        const int k_base = tile_k * BK;

        // A tile load: 각 thread가 1 element
        int a_row = row;
        int a_col = k_base + tx;

        if (a_row < M && a_col < K) {
            As[ty][tx] = A[a_row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // B tile load: 각 thread가 TN element
#pragma unroll
        for (int t = 0; t < TN; ++t) {
            int b_col = col0 + t;
            int b_row = k_base + ty;

            if (b_row < K && b_col < N) {
                Bs[ty][tx * TN + t] = B[b_row * N + b_col];
            } else {
                Bs[ty][tx * TN + t] = 0.0f;
            }
        }

        __syncthreads();

        // tile GEMM 내부 루프 (K 축 BK)
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a_val = As[ty][kk];

#pragma unroll
            for (int t = 0; t < TN; ++t) {
                float b_val = Bs[kk][tx * TN + t];
                acc[t] = fmaf(a_val, b_val, acc[t]);
            }
        }

        __syncthreads();
    }

    // 결과 쓰기
    if (row < M) {
#pragma unroll
        for (int t = 0; t < TN; ++t) {
            int col = col0 + t;
            if (col < N) {
                C[row * N + col] = acc[t];
            }
        }
    }
}

// =======================================
// host helper
// =======================================

template<int TN>
void run_config(const float* dA, const float* dB,
                float* dC, const float* hC_ref, float* hC_out)
{
    constexpr int BN = THREADS_X * TN;

    dim3 block(THREADS_X, THREADS_Y);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    // warm-up
    gemm_multi_acc_TN_kernel<TN><<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    gemm_multi_acc_TN_kernel<TN><<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaMemcpy(hC_out, dC, M * N * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float max_diff = max_abs_diff(hC_ref, hC_out, M * N);

    printf("Config TN=%d  BN=%3d  |  time=%.3f ms  |  max diff=%.9e\n",
           TN, BN, ms, max_diff);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// =======================================
// main
// =======================================

int main()
{
    printf("=== Multi-Accumulator GEMM Test: TN sweep (1,2,4,8) ===\n");
    printf("GEMM config: C[%d x %d] = A[%d x %d] * B[%d x %d]\n",
           M, N, M, K, K, N);
    printf("Block tile: BM=%d, BK=%d, BN=16*TN, BlockDim=(%d,%d)\n\n",
           BM, BK, THREADS_X, THREADS_Y);

    // host alloc
    float* hA     = (float*)malloc(M * K * sizeof(float));
    float* hB     = (float*)malloc(K * N * sizeof(float));
    float* hC_ref = (float*)malloc(M * N * sizeof(float));
    float* hC_out = (float*)malloc(M * N * sizeof(float));

    // init A,B
    for (int i = 0; i < M * K; ++i) {
        hA[i] = (float)((i % 17) - 8) * 0.1f;
    }
    for (int i = 0; i < K * N; ++i) {
        hB[i] = (float)((i % 13) - 6) * 0.05f;
    }

    printf("Computing host reference...\n");
    host_gemm_ref(hA, hB, hC_ref, M, N, K);

    // device alloc
    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, K * N * sizeof(float), cudaMemcpyHostToDevice));

    printf("\nRunning TN configs...\n");
    run_config<1>(dA, dB, dC, hC_ref, hC_out);
    run_config<2>(dA, dB, dC, hC_ref, hC_out);
    run_config<4>(dA, dB, dC, hC_ref, hC_out);
    run_config<8>(dA, dB, dC, hC_ref, hC_out);

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    free(hA);
    free(hB);
    free(hC_ref);
    free(hC_out);

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}
/*
nvcc -O3 -arch=sm_86 -lineinfo -Xptxas=-v -o gemm_multi_acc_TN_test.exe gemm_multi_acc_TN_test.cu

ncu --kernel-name regex:gemm_multi_acc_TN_kernel.* --metrics smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread ./gemm_multi_acc_TN_test.exe

*/