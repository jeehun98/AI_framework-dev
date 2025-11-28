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
// GEMM 설정
// =======================
// C[M x N] = A[M x K] * B[K x N]
constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;

// 타일 크기
constexpr int BM = 16;   // block이 담당하는 M 타일
constexpr int BK = 16;   // K 타일 두께
// BN = 16 * TN (템플릿에서 결정)

// 스레드 블록 크기 (고정)
constexpr int TX = 16;
constexpr int TY = 16;

// =======================
// 유틸 함수
// =======================
void init_matrix(float* mat, int rows, int cols, float scale = 1.0f) {
    for (int i = 0; i < rows * cols; ++i) {
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

// host-side GEMM (검증용)
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
// Register Tiling GEMM Kernel (TN 방향)
// =======================
//
//  - Block tile: C[BM x BN] = [16 x (16*TN)]
//  - blockDim = (TX=16, TY=16)
//  - thread당 N 방향 결과: acc[TN]
//  - TN↑ → 레지스터 사용량↑ → occupancy↓ 예상
//
template<int TN>
__global__ void gemm_reg_tiling_TN_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    constexpr int BN = 16 * TN;
    static_assert(BM == 16, "BM must be 16");
    static_assert(BK == 16, "BK must be 16");

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = block_row + ty;
    int col_base = block_col + tx;

    __shared__ float As[BM][BK];    // [16 x 16]
    __shared__ float Bs[BK][BN];    // [16 x (16*TN)]

    // TN 개의 결과를 레지스터에 보관 (register pressure 증가 포인트)
    float acc[TN];
    #pragma unroll
    for (int i = 0; i < TN; ++i) {
        acc[i] = 0.0f;
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        // A tile load
        for (int i = ty; i < BM; i += TY) {
            int g_row = block_row + i;
            for (int j = tx; j < BK; j += TX) {
                int g_k = k0 + j;
                float val = 0.0f;
                if (g_row < M && g_k < K) {
                    val = A[g_row * K + g_k];
                }
                As[i][j] = val;
            }
        }

        // B tile load
        for (int i = ty; i < BK; i += TY) {
            int g_k = k0 + i;
            for (int j = tx; j < BN; j += TX) {
                int g_col = block_col + j;
                float val = 0.0f;
                if (g_k < K && g_col < N) {
                    val = B[g_k * N + g_col];
                }
                Bs[i][j] = val;
            }
        }

        __syncthreads();

        // compute
        if (row < M) {
            #pragma unroll
            for (int kk = 0; kk < BK; ++kk) {
                float a_val = As[ty][kk];

                #pragma unroll
                for (int t = 0; t < TN; ++t) {
                    int col = col_base + t * TX;
                    float b_val = 0.0f;
                    if (col < N) {
                        int b_col_in_tile = tx + t * TX;
                        b_val = Bs[kk][b_col_in_tile];
                    }
                    acc[t] += a_val * b_val;
                }
            }
        }

        __syncthreads();
    }

    if (row < M) {
        #pragma unroll
        for (int t = 0; t < TN; ++t) {
            int col = col_base + t * TX;
            if (col < N) {
                C[row * N + col] = acc[t];
            }
        }
    }
}

// =======================
// 설정별 실행 헬퍼
// =======================
template<int TN>
void run_config(const char* name,
                const float* dA, const float* dB, float* dC,
                const float* hC_ref, float* hC_tmp)
{
    constexpr int BN = 16 * TN;

    dim3 block(TX, TY);  // (16,16)
    dim3 grid((N + BN - 1) / BN,
              (M + BM - 1) / BM);

    size_t smem_bytes =
        sizeof(float) * (BM * BK + BK * BN);
    double smem_kb = smem_bytes / 1024.0;

    // warm-up
    gemm_reg_tiling_TN_kernel<TN><<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    gemm_reg_tiling_TN_kernel<TN><<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaMemcpy(hC_tmp, dC,
                          sizeof(float) * M * N,
                          cudaMemcpyDeviceToHost));
    float diff = max_abs_diff(hC_ref, hC_tmp, M * N);

    printf("Config %-8s  TN=%2d  BN=%3d  |  smem=%.1f KB  |  time=%.3f ms  |  max diff=%e\n",
           name, TN, BN, smem_kb, ms, diff);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// =======================
// main: TN 스윕
// =======================
//
//  Test 1: TN = 1, 2, 4, 8
//  Test 2(occupancy): TN = 8, 12, 16 쪽이 register 압박 더 큼
//
int main()
{
    printf("=== Register Tiling Test 2: Occupancy vs Performance ===\n");
    printf("GEMM config: C[%d x %d] = A[%d x %d] * B[%d x %d]\n",
           M, N, M, K, K, N);
    printf("Block tile: BM=%d, BK=%d, BN=16*TN\n", BM, BK);
    printf("BlockDim:   (%d, %d)\n\n", TX, TY);

    size_t bytesA = sizeof(float) * M * K;
    size_t bytesB = sizeof(float) * K * N;
    size_t bytesC = sizeof(float) * M * N;

    float *hA      = (float*)malloc(bytesA);
    float *hB      = (float*)malloc(bytesB);
    float *hC_ref  = (float*)malloc(bytesC);
    float *hC_tmp  = (float*)malloc(bytesC);

    init_matrix(hA, M, K, 0.01f);
    init_matrix(hB, K, N, 0.02f);

    printf("Computing host reference...\n");
    gemm_host_ref(hA, hB, hC_ref, M, N, K);

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));

    CHECK_CUDA(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

    printf("\nRunning TN configs (increasing register pressure)...\n");

    // 낮은 register pressure
    run_config<1>("TN1",  dA, dB, dC, hC_ref, hC_tmp);
    run_config<2>("TN2",  dA, dB, dC, hC_ref, hC_tmp);
    run_config<4>("TN4",  dA, dB, dC, hC_ref, hC_tmp);
    run_config<8>("TN8",  dA, dB, dC, hC_ref, hC_tmp);

    // 더 공격적인 TN (occupancy 감소 유도)
    run_config<12>("TN12", dA, dB, dC, hC_ref, hC_tmp);
    run_config<16>("TN16", dA, dB, dC, hC_ref, hC_tmp);

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    free(hA);
    free(hB);
    free(hC_ref);
    free(hC_tmp);

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}

/*
nvcc -O3 -arch=sm_86 -lineinfo -o gemm_register_occupancy_test.exe gemm_register_occupancy_test.cu


# TN=1
ncu --kernel-name regex:gemm_reg_tiling_TN_kernel.* --metrics sm__warps_active.avg,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed C:\Users\owner\Desktop\AI_framework-dev\GPU_test\02_Optimization_Principles\2_2_register_tiling\gemm_register_occupancy_test.exe
*/