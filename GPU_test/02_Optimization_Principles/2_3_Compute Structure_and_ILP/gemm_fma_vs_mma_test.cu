// gemm_fma_vs_mma_test.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t status_ = (call);                                           \
        if (status_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Error %s:%d: %s\n",                           \
                    __FILE__, __LINE__, cudaGetErrorString(status_));           \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)
#endif

// GEMM config: C[M x N] = A[M x K] * B[K x N]
constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;

// -------------------------
// FP32 FMA GEMM kernel
//  - shared tiling, 16x16 tile
// -------------------------
constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 16;

__global__ void gemm_fma_fp32_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int N, int K)
{
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    float acc = 0.0f;

    for (int kt = 0; kt < K; kt += TILE_K) {
        // load tile A (row-major)
        if (row < M && (kt + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + (kt + threadIdx.x)];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // load tile B (row-major)
        if ((kt + threadIdx.y) < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(kt + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // compute
        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            acc = fmaf(As[threadIdx.y][kk], Bs[kk][threadIdx.x], acc);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// -------------------------
// FP16 MMA (Tensor Core) GEMM kernel (WMMA)
//  - A: row-major, B: col-major
//  - each warp computes a 16x16 tile
// -------------------------
__global__ void gemm_mma_fp16_kernel(const half* __restrict__ A,
                                     const half* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int N, int K)
{
#if __CUDA_ARCH__ >= 700
    // 1 warp = 16x16 tile of C
    int warpId  = (threadIdx.x / 32); // warps per block in x-dim only
    int laneId  = threadIdx.x % 32;

    // block covers multiple 16x16 tiles in M dimension if blockDim.x > 32
    // 여기선 blockDim.x=32, blockDim.y >=1 로 가정해서 warpId==threadIdx.y 라고 둬도 됨
    // 간단하게: blockDim.x == 32, blockDim.y == 1 가정.
    int warpM = blockIdx.y; // tile index in M (16 rows per warp)
    int warpN = blockIdx.x; // tile index in N (16 cols per warp)

    if (warpId > 0) return; // block당 1 warp만 사용 (단순화용)

    int m0 = warpM * 16;
    int n0 = warpN * 16;

    if (m0 >= M || n0 >= N) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>   a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>   b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>                c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // loop over K in steps of 16
    for (int k0 = 0; k0 < K; k0 += 16) {
        const half* tileA = A + m0 * K + k0;       // row-major A
        const half* tileB = B + n0 * K + k0;       // col-major B (ld = K)

        wmma::load_matrix_sync(a_frag, tileA, K);
        wmma::load_matrix_sync(b_frag, tileB, K);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // store
    float* tileC = C + m0 * N + n0;
    wmma::store_matrix_sync(tileC, c_frag, N, wmma::mem_row_major);
#else
    // Tensor Core 없음
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
#endif
}

// -------------------------
// Host util
// -------------------------
double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    return static_cast<double>(ms);
}

int main()
{
    std::printf("=== MMA / WMMA Test 1: FP32 FMA GEMM vs FP16 MMA GEMM ===\n");
    std::printf("GEMM config: C[%d x %d] = A[%d x %d] * B[%d x %d]\n\n",
                M, N, M, K, K, N);

    size_t bytes_A_f32 = M * K * sizeof(float);
    size_t bytes_B_f32 = K * N * sizeof(float);
    size_t bytes_C_f32 = M * N * sizeof(float);
    size_t bytes_A_f16 = M * K * sizeof(half);
    size_t bytes_B_f16 = K * N * sizeof(half);

    // host alloc
    float* hA_f32 = (float*)std::malloc(bytes_A_f32);
    float* hB_f32 = (float*)std::malloc(bytes_B_f32);
    float* hC_fma = (float*)std::malloc(bytes_C_f32);
    float* hC_mma = (float*)std::malloc(bytes_C_f32);

    if (!hA_f32 || !hB_f32 || !hC_fma || !hC_mma) {
        std::fprintf(stderr, "Host malloc failed\n");
        return EXIT_FAILURE;
    }

    // init input
    std::srand(42);
    for (int i = 0; i < M * K; ++i) {
        hA_f32[i] = (std::rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
    for (int i = 0; i < K * N; ++i) {
        hB_f32[i] = (std::rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    // device alloc
    float *dA_f32, *dB_f32, *dC_fma, *dC_mma;
    half  *dA_f16, *dB_f16;

    CHECK_CUDA(cudaMalloc(&dA_f32, bytes_A_f32));
    CHECK_CUDA(cudaMalloc(&dB_f32, bytes_B_f32));
    CHECK_CUDA(cudaMalloc(&dC_fma, bytes_C_f32));
    CHECK_CUDA(cudaMalloc(&dC_mma, bytes_C_f32));
    CHECK_CUDA(cudaMalloc(&dA_f16, bytes_A_f16));
    CHECK_CUDA(cudaMalloc(&dB_f16, bytes_B_f16));

    CHECK_CUDA(cudaMemcpy(dA_f32, hA_f32, bytes_A_f32, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_f32, hB_f32, bytes_B_f32, cudaMemcpyHostToDevice));

    // FP16 copy (A row-major, B col-major for MMA)
    // A_fp16: just cast row-major
    // B_fp16: store as col-major (so we transpose KxN)
    half* hA_f16 = (half*)std::malloc(bytes_A_f16);
    half* hB_f16 = (half*)std::malloc(bytes_B_f16);
    if (!hA_f16 || !hB_f16) {
        std::fprintf(stderr, "Host malloc (half) failed\n");
        return EXIT_FAILURE;
    }

    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            hA_f16[m * K + k] = __float2half(hA_f32[m * K + k]);
        }
    }
    // B: row-major (K x N) -> col-major (N x K)
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            float v = hB_f32[k * N + n];
            hB_f16[n * K + k] = __float2half(v);   // (col-major)
        }
    }

    CHECK_CUDA(cudaMemcpy(dA_f16, hA_f16, bytes_A_f16, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_f16, hB_f16, bytes_B_f16, cudaMemcpyHostToDevice));

    // -------------------------
    // FP32 FMA kernel
    // -------------------------
    dim3 block_fma(TILE_N, TILE_M); // (16,16)
    dim3 grid_fma((N + TILE_N - 1) / TILE_N,
                  (M + TILE_M - 1) / TILE_M);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::printf("[FMA] warm-up + timing\n");
    // warm-up
    gemm_fma_fp32_kernel<<<grid_fma, block_fma>>>(dA_f32, dB_f32, dC_fma, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    gemm_fma_fp32_kernel<<<grid_fma, block_fma>>>(dA_f32, dB_f32, dC_fma, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    double fma_ms = elapsed_ms(start, stop);

    // -------------------------
    // FP16 MMA (WMMA) kernel
    // -------------------------
    // block당 1 warp만 사용: blockDim.x = 32, blockDim.y = 1
    dim3 block_mma(32, 1);
    dim3 grid_mma((N + 16 - 1) / 16,
                  (M + 16 - 1) / 16);

    std::printf("\n[MMA] warm-up + timing\n");
    // warm-up
    gemm_mma_fp16_kernel<<<grid_mma, block_mma>>>(dA_f16, dB_f16, dC_mma, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    gemm_mma_fp16_kernel<<<grid_mma, block_mma>>>(dA_f16, dB_f16, dC_mma, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    double mma_ms = elapsed_ms(start, stop);

    // -------------------------
    // 결과 수집 + 비교
    // -------------------------
    CHECK_CUDA(cudaMemcpy(hC_fma, dC_fma, bytes_C_f32, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_mma, dC_mma, bytes_C_f32, cudaMemcpyDeviceToHost));

    double max_diff = 0.0;
    for (int i = 0; i < M * N; ++i) {
        double diff = std::fabs((double)hC_fma[i] - (double)hC_mma[i]);
        if (diff > max_diff) max_diff = diff;
    }

    // FLOP 계산 (FMA 1회 = 2 FLOPs)
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double fma_gflops = flops / (fma_ms * 1e-3) / 1e9;
    double mma_gflops = flops / (mma_ms * 1e-3) / 1e9;

    std::printf("\n=== Results ===\n");
    std::printf("[FMA] time = %.3f ms  |  GFLOPs = %.2f\n", fma_ms, fma_gflops);
    std::printf("[MMA] time = %.3f ms  |  GFLOPs = %.2f\n", mma_ms, mma_gflops);
    std::printf("max |C_fma - C_mma| = %.6e\n", max_diff);

    // cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    cudaFree(dA_f32);
    cudaFree(dB_f32);
    cudaFree(dC_fma);
    cudaFree(dC_mma);
    cudaFree(dA_f16);
    cudaFree(dB_f16);

    std::free(hA_f32);
    std::free(hB_f32);
    std::free(hC_fma);
    std::free(hC_mma);
    std::free(hA_f16);
    std::free(hB_f16);

    return 0;
}
/*
nvcc -O3 -arch=sm_86 gemm_fma_vs_mma_test.cu -o gemm_fma_vs_mma_test.exe
# FP32 FMA kernel
ncu --kernel-name regex:gemm_fma_fp32_kernel.* --metrics smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed ./gemm_fma_vs_mma_test.exe

# FP16 MMA kernel (Tensor Core)
ncu --kernel-name regex:gemm_mma_fp16_kernel.* --metrics smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed ./gemm_fma_vs_mma_test.exe

*/