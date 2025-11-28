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

// 고정 thread tile 크기
constexpr int TM = 16; // blockDim.y
constexpr int TN = 16; // blockDim.x

// =======================
// 유틸 함수들
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

// 단순 host-side GEMM (검증용)
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
// Templated Tiled GEMM Kernel
// =======================
//
// BM, BK, BN: tile size
// - block이 C의 [BM x BN] 영역 담당
// - K 방향은 BK 단위로 쪼개서 loop
//
// blockDim = (TN, TM) = (16, 16)
// thread 당 C micro-tile 크기:
//   RM = BM / TM, RN = BN / TN
//
template<int BM, int BK, int BN>
__global__ void gemm_smem_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    static_assert(BM % TM == 0, "BM must be multiple of TM");
    static_assert(BN % TN == 0, "BN must be multiple of TN");

    constexpr int RM = BM / TM;
    constexpr int RN = BN / TN;

    // block이 담당하는 C 타일의 시작 좌표
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // thread가 담당하는 C micro-tile 시작 좌표
    int row_base = block_row + threadIdx.y * RM;
    int col_base = block_col + threadIdx.x * RN;

    // shared tiles
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // 레지스터에 누적하는 C micro-tile
    float acc[RM][RN];

    #pragma unroll
    for (int i = 0; i < RM; ++i) {
        #pragma unroll
        for (int j = 0; j < RN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // K 방향을 BK 단위로 타일링
    for (int k0 = 0; k0 < K; k0 += BK) {
        // =======================
        // 1) shared memory 로드
        // =======================
        // A tile: [BM x BK]
        for (int i = threadIdx.y; i < BM; i += blockDim.y) {
            int global_row = block_row + i;
            for (int j = threadIdx.x; j < BK; j += blockDim.x) {
                int global_k = k0 + j;
                float val = 0.0f;
                if (global_row < M && global_k < K) {
                    val = A[global_row * K + global_k];
                }
                As[i][j] = val;
            }
        }

        // B tile: [BK x BN]
        for (int i = threadIdx.y; i < BK; i += blockDim.y) {
            int global_k = k0 + i;
            for (int j = threadIdx.x; j < BN; j += blockDim.x) {
                int global_col = block_col + j;
                float val = 0.0f;
                if (global_k < K && global_col < N) {
                    val = B[global_k * N + global_col];
                }
                Bs[i][j] = val;
            }
        }

        __syncthreads();

        // =======================
        // 2) tile 내부 연산
        // =======================
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            // thread가 담당하는 RM x RN 영역에 대해
            float a_frag[RM];
            float b_frag[RN];

            // A fragment: (RM rows)
            #pragma unroll
            for (int i = 0; i < RM; ++i) {
                int r = threadIdx.y * RM + i; // [0, BM)
                a_frag[i] = As[r][kk];
            }

            // B fragment: (RN cols)
            #pragma unroll
            for (int j = 0; j < RN; ++j) {
                int c = threadIdx.x * RN + j; // [0, BN)
                b_frag[j] = Bs[kk][c];
            }

            // outer product
            #pragma unroll
            for (int i = 0; i < RM; ++i) {
                float a_val = a_frag[i];
                #pragma unroll
                for (int j = 0; j < RN; ++j) {
                    acc[i][j] += a_val * b_frag[j];
                }
            }
        }

        __syncthreads();
    }

    // =======================
    // 3) 결과 쓰기
    // =======================
    #pragma unroll
    for (int i = 0; i < RM; ++i) {
        int row = row_base + i;
        if (row >= M) continue;
        #pragma unroll
        for (int j = 0; j < RN; ++j) {
            int col = col_base + j;
            if (col >= N) continue;
            C[row * N + col] = acc[i][j];
        }
    }
}

// =======================
// 특정 (BM,BK,BN) 조합 실행 헬퍼
// =======================
template<int BM, int BK, int BN>
void run_config(const char* name,
                const float* dA, const float* dB, float* dC,
                const float* hC_ref, float* hC_tmp)
{
    dim3 block(TN, TM); // (16,16)
    dim3 grid((N + BN - 1) / BN,
              (M + BM - 1) / BM);

    // shared memory 사용량
    constexpr size_t smem_bytes =
        sizeof(float) * (BM * BK + BK * BN);
    double smem_kb = smem_bytes / 1024.0;

    // warm-up
    gemm_smem_tiled_kernel<BM, BK, BN><<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    gemm_smem_tiled_kernel<BM, BK, BN><<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaMemcpy(hC_tmp, dC,
                          sizeof(float) * M * N,
                          cudaMemcpyDeviceToHost));
    float diff = max_abs_diff(hC_ref, hC_tmp, M * N);

    printf("Config %-15s  BM=%3d BK=%2d BN=%3d  |  smem=%.1f KB  |  time=%.3f ms  |  max diff=%e\n",
           name, BM, BK, BN, smem_kb, ms, diff);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// =======================
// main: (BM,BK,BN) sweep
// =======================
int main() {
    printf("=== Test 2: SMEM tile size sweep (BM, BK, BN) ===\n");
    printf("GEMM config: C[%d x %d] = A[%d x %d] * B[%d x %d]\n",
           M, N, M, K, K, N);
    printf("Thread tile: TM=%d, TN=%d (blockDim=(%d,%d))\n",
           TM, TN, TN, TM);

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

    printf("\nRunning configs...\n");

    // BM = 32, BK = 8/16/32, BN = 32/64/128
    run_config<32,  8,  32>("BM32_BK8_BN32",  dA, dB, dC, hC_ref, hC_tmp);
    run_config<32,  8,  64>("BM32_BK8_BN64",  dA, dB, dC, hC_ref, hC_tmp);
    run_config<32,  8, 128>("BM32_BK8_BN128", dA, dB, dC, hC_ref, hC_tmp);

    run_config<32, 16,  32>("BM32_BK16_BN32",  dA, dB, dC, hC_ref, hC_tmp);
    run_config<32, 16,  64>("BM32_BK16_BN64",  dA, dB, dC, hC_ref, hC_tmp);
    run_config<32, 16, 128>("BM32_BK16_BN128", dA, dB, dC, hC_ref, hC_tmp);

    run_config<32, 32,  32>("BM32_BK32_BN32",  dA, dB, dC, hC_ref, hC_tmp);
    run_config<32, 32,  64>("BM32_BK32_BN64",  dA, dB, dC, hC_ref, hC_tmp);
    run_config<32, 32, 128>("BM32_BK32_BN128", dA, dB, dC, hC_ref, hC_tmp);

    // BM = 64
    run_config<64,  8,  32>("BM64_BK8_BN32",  dA, dB, dC, hC_ref, hC_tmp);
    run_config<64,  8,  64>("BM64_BK8_BN64",  dA, dB, dC, hC_ref, hC_tmp);
    run_config<64,  8, 128>("BM64_BK8_BN128", dA, dB, dC, hC_ref, hC_tmp);

    run_config<64, 16,  32>("BM64_BK16_BN32",  dA, dB, dC, hC_ref, hC_tmp);
    run_config<64, 16,  64>("BM64_BK16_BN64",  dA, dB, dC, hC_ref, hC_tmp);
    run_config<64, 16, 128>("BM64_BK16_BN128", dA, dB, dC, hC_ref, hC_tmp);

    run_config<64, 32,  32>("BM64_BK32_BN32",  dA, dB, dC, hC_ref, hC_tmp);
    run_config<64, 32,  64>("BM64_BK32_BN64",  dA, dB, dC, hC_ref, hC_tmp);
    run_config<64, 32, 128>("BM64_BK32_BN128", dA, dB, dC, hC_ref, hC_tmp);

    // BM = 128
    run_config<128,  8,  32>("BM128_BK8_BN32",  dA, dB, dC, hC_ref, hC_tmp);
    run_config<128,  8,  64>("BM128_BK8_BN64",  dA, dB, dC, hC_ref, hC_tmp);
    run_config<128,  8, 128>("BM128_BK8_BN128", dA, dB, dC, hC_ref, hC_tmp);

    run_config<128, 16,  32>("BM128_BK16_BN32",  dA, dB, dC, hC_ref, hC_tmp);
    run_config<128, 16,  64>("BM128_BK16_BN64",  dA, dB, dC, hC_ref, hC_tmp);
    run_config<128, 16, 128>("BM128_BK16_BN128", dA, dB, dC, hC_ref, hC_tmp);

    run_config<128, 32,  32>("BM128_BK32_BN32",  dA, dB, dC, hC_ref, hC_tmp);
    run_config<128, 32,  64>("BM128_BK32_BN64",  dA, dB, dC, hC_ref, hC_tmp);
    run_config<128, 32, 128>("BM128_BK32_BN128", dA, dB, dC, hC_ref, hC_tmp);

    // cleanup
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

// nvcc -O3 -arch=sm_86 -o gemm_tiling_sweep.exe gemm_tiling_sweep.cu

