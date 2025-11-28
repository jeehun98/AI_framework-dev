// gemm_mma_warp_tile_test.cu
// Warp-level Tiling Test 3: FMA vs MMA (Tensor Core)
//
// C[M x N] = A[M x K] * B[K x N]
// A, B: FP16, C: FP32 accumulate
// 비교: gemm_warp_fma_kernel vs gemm_warp_mma_kernel

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

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

constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;

constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;

// =======================
// 유틸 함수
// =======================

void init_matrix_f32(float* mat, int rows, int cols, float scale = 1.0f) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = scale * ((i % 13) - 6);
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

// host GEMM: A row-major, B col-major
void gemm_host_ref(const float* A, const float* B_col, float* C,
                   int M, int N, int K)
{
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a = A[row * K + k];
                float b = B_col[col * K + k];  // col-major
                acc += a * b;
            }
            C[row * N + col] = acc;
        }
    }
}

void convert_f32_to_f16_row(const float* src, __half* dst,
                            int rows, int cols)
{
    int n = rows * cols;
    for (int i = 0; i < n; ++i) {
        dst[i] = __float2half(src[i]);
    }
}

void convert_f32_rm_to_f16_cm(const float* src_rm, __half* dst_cm,
                              int rows, int cols)
{
    // src_rm: row-major [rows x cols]
    // dst_cm: col-major [rows x cols]
    // 여기선 rows=K, cols=N
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float v = src_rm[r * cols + c];
            dst_cm[c * rows + r] = __float2half(v);
        }
    }
}

// ================================
// FMA 기반 warp-level GEMM
// ================================

__global__ void gemm_warp_fma_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ Bc,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int lane = threadIdx.x; // 0..31

    int tile_m = blockIdx.y * WM;
    int tile_n = blockIdx.x * WN;

    __shared__ float As[WM][WK]; // 16x16
    __shared__ float Bs[WK][WN]; // 16x16

    int row_group = lane % 8;   // 0..7
    int col_group = lane / 8;   // 0..3

    int row0 = tile_m + row_group * 2 + 0;
    int row1 = tile_m + row_group * 2 + 1;
    int col0 = tile_n + col_group * 4;

    float acc[2][4];
    #pragma unroll
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 4; ++c)
            acc[r][c] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += WK) {
        // load A tile: [16 x 16], row-major
        for (int idx = lane; idx < WM * WK; idx += 32) {
            int r = idx / WK;
            int c = idx % WK;
            int g_row = tile_m + r;
            int g_col = k0 + c;
            float v = 0.0f;
            if (g_row < M && g_col < K) {
                __half h = A[g_row * K + g_col];
                v = __half2float(h);
            }
            As[r][c] = v;
        }

        // load B tile: [16 x 16], col-major
        for (int idx = lane; idx < WK * WN; idx += 32) {
            int r = idx / WN;
            int c = idx % WN;
            int g_k = k0 + r;
            int g_n = tile_n + c;
            float v = 0.0f;
            if (g_k < K && g_n < N) {
                __half h = Bc[g_n * K + g_k];
                v = __half2float(h);
            }
            Bs[r][c] = v;
        }

        __syncwarp();

        // compute 2x4 per lane
        #pragma unroll
        for (int kk = 0; kk < WK; ++kk) {
            float a0 = 0.0f;
            float a1 = 0.0f;

            if (row0 < M) a0 = As[row0 - tile_m][kk];
            if (row1 < M) a1 = As[row1 - tile_m][kk];

            float b[4];
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                int col = col0 + c;
                if (col < N)
                    b[c] = Bs[kk][col - tile_n];
                else
                    b[c] = 0.0f;
            }

            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                acc[0][c] += a0 * b[c];
                acc[1][c] += a1 * b[c];
            }
        }

        __syncwarp();
    }

    if (row0 < M) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            int col = col0 + c;
            if (col < N)
                C[row0 * N + col] = acc[0][c];
        }
    }
    if (row1 < M) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            int col = col0 + c;
            if (col < N)
                C[row1 * N + col] = acc[1][c];
        }
    }
}

// ================================
// MMA 기반 warp-level GEMM (Tensor Core)
// ================================

__global__ void gemm_warp_mma_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ Bc,
    float* __restrict__ C,
    int M, int N, int K)
{
#if __CUDA_ARCH__ < 700
    return;
#endif
    const int warpId = threadIdx.x / 32;
    if (warpId > 0) return;

    int tile_m = blockIdx.y * WM;
    int tile_n = blockIdx.x * WN;

    wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += WK) {
        const __half* tileA = A  + tile_m * K + k0;
        const __half* tileB = Bc + tile_n * K + k0;
        wmma::load_matrix_sync(a_frag, tileA, K);
        wmma::load_matrix_sync(b_frag, tileB, K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    if (tile_m < M && tile_n < N) {
        float* tileC = C + tile_m * N + tile_n;
        wmma::store_matrix_sync(tileC, c_frag, N, wmma::mem_row_major);
    }
}

// =======================
// main
// =======================

int main()
{
    printf("=== Warp-level Tiling Test 3: FMA vs MMA (Tensor Core) ===\n");
    printf("GEMM config: C[%d x %d] = A[%d x %d] * B[%d x %d]\n",
           M, N, M, K, K, N);
    printf("Tile: WM=%d, WN=%d, WK=%d\n\n", WM, WN, WK);

    size_t bytesA_f32 = sizeof(float) * M * K;
    size_t bytesB_f32 = sizeof(float) * K * N;
    size_t bytesC_f32 = sizeof(float) * M * N;
    size_t bytesA_f16 = sizeof(__half) * M * K;
    size_t bytesB_f16 = sizeof(__half) * K * N;

    float *hA_f32     = (float*)malloc(bytesA_f32);
    float *hB_f32_rm  = (float*)malloc(bytesB_f32);
    float *hC_ref     = (float*)malloc(bytesC_f32);
    float *hC_fma     = (float*)malloc(bytesC_f32);
    float *hC_mma     = (float*)malloc(bytesC_f32);

    __half *hA_f16    = (__half*)malloc(bytesA_f16);
    __half *hB_f16_cm = (__half*)malloc(bytesB_f16);

    init_matrix_f32(hA_f32,    M, K, 0.01f);
    init_matrix_f32(hB_f32_rm, K, N, 0.02f);

    printf("Computing host reference (FP32, A row-major, B col-major)...\n");
    {
        float* hB_f32_cm = (float*)malloc(bytesB_f32);
        for (int k = 0; k < K; ++k) {
            for (int n = 0; n < N; ++n) {
                hB_f32_cm[n * K + k] = hB_f32_rm[k * N + n];
            }
        }
        gemm_host_ref(hA_f32, hB_f32_cm, hC_ref, M, N, K);
        free(hB_f32_cm);
    }

    convert_f32_to_f16_row(hA_f32, hA_f16, M, K);
    convert_f32_rm_to_f16_cm(hB_f32_rm, hB_f16_cm, K, N);

    __half *dA_f16, *dB_f16_cm;
    float  *dC_fma, *dC_mma;

    CHECK_CUDA(cudaMalloc(&dA_f16,    bytesA_f16));
    CHECK_CUDA(cudaMalloc(&dB_f16_cm, bytesB_f16));
    CHECK_CUDA(cudaMalloc(&dC_fma,    bytesC_f32));
    CHECK_CUDA(cudaMalloc(&dC_mma,    bytesC_f32));

    CHECK_CUDA(cudaMemcpy(dA_f16,    hA_f16,    bytesA_f16, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_f16_cm, hB_f16_cm, bytesB_f16, cudaMemcpyHostToDevice));

    dim3 grid((N + WN - 1) / WN,
              (M + WM - 1) / WM);
    dim3 block_warp(32, 1);

    // FMA warp-tile
    {
        printf("\n[FMA warp-tile] warm-up + timing\n");
        gemm_warp_fma_kernel<<<grid, block_warp>>>(dA_f16, dB_f16_cm, dC_fma, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        gemm_warp_fma_kernel<<<grid, block_warp>>>(dA_f16, dB_f16_cm, dC_fma, M, N, K);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        CHECK_CUDA(cudaMemcpy(hC_fma, dC_fma, bytesC_f32, cudaMemcpyDeviceToHost));
        float diff = max_abs_diff(hC_ref, hC_fma, M * N);

        printf("[FMA warp-tile]  time: %.3f ms, max diff vs ref = %e\n", ms, diff);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    // MMA warp-tile
    {
        printf("\n[MMA warp-tile] warm-up + timing\n");
        gemm_warp_mma_kernel<<<grid, block_warp>>>(dA_f16, dB_f16_cm, dC_mma, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        gemm_warp_mma_kernel<<<grid, block_warp>>>(dA_f16, dB_f16_cm, dC_mma, M, N, K);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        CHECK_CUDA(cudaMemcpy(hC_mma, dC_mma, bytesC_f32, cudaMemcpyDeviceToHost));
        float diff = max_abs_diff(hC_ref, hC_mma, M * N);

        printf("[MMA warp-tile] time: %.3f ms, max diff vs ref = %e\n", ms, diff);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    CHECK_CUDA(cudaFree(dA_f16));
    CHECK_CUDA(cudaFree(dB_f16_cm));
    CHECK_CUDA(cudaFree(dC_fma));
    CHECK_CUDA(cudaFree(dC_mma));

    free(hA_f32);
    free(hB_f32_rm);
    free(hC_ref);
    free(hC_fma);
    free(hC_mma);
    free(hA_f16);
    free(hB_f16_cm);

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}

/*
nvcc -O3 -arch=sm_86 -lineinfo -o gemm_mma_warp_tile_test.exe gemm_mma_warp_tile_test.cu

ncu --kernel-name regex:gemm_warp_fma_kernel.* --metrics smsp__inst_executed.sum,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed ./gemm_mma_warp_tile_test.exe
ncu --kernel-name regex:gemm_warp_mma_kernel.* --metrics smsp__inst_executed.sum,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed ./gemm_mma_warp_tile_test.exe

*/