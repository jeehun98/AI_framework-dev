// test_tc_gemm.cu
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err__));            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
#endif

// ----------------------
// FP32 GEMM (naive/row-major)
// C[M,N] = A[M,K] * B[K,N]
// ----------------------
__global__ void fp32_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // [0,M)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // [0,N)

    if (row >= M || col >= N) return;

    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

// ----------------------
// Tensor Core GEMM (WMMA, m16n16k16)
// A, B: half, row-major
// C: float, row-major
// ----------------------
__global__ void wmma_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // 한 타일 = 16x16 출력
    int tileRow = blockIdx.y; // tile row index
    int tileCol = blockIdx.x; // tile col index

    int row = tileRow * 16;
    int col = tileCol * 16;

    if (row >= M || col >= N) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator,16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // K dimension을 16씩 잘라서 accumulate
    for (int k = 0; k < K; k += 16) {
        const half* a_tile = A + (row * K + k);
        const half* b_tile = B + (k * N + col);

        wmma::load_matrix_sync(a_frag, a_tile, K);
        wmma::load_matrix_sync(b_frag, b_tile, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 결과 저장
    float* c_tile = C + (row * N + col);
    wmma::store_matrix_sync(c_tile, c_frag, N, wmma::mem_row_major);
}

// ----------------------
// Helper: 초기화 & timing
// ----------------------
void init_host_float(float* ptr, int n) {
    for (int i = 0; i < n; ++i) {
        float v = (std::rand() / (float)RAND_MAX) * 2.f - 1.f;
        ptr[i] = v;
    }
}

void float_to_half(const float* src, half* dst, int n) {
    for (int i = 0; i < n; ++i) {
        dst[i] = __float2half(src[i]);
    }
}

float run_fp32_gemm(int M, int N, int K,
                    const float* dA, const float* dB, float* dC)
{
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // warmup
    fp32_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    fp32_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms;
}

float run_wmma_gemm(int M, int N, int K,
                    const half* dA, const half* dB, float* dC)
{
    // 한 블록=한 warp 라고 가정 (32 threads)
    dim3 block(32, 1, 1);
    dim3 grid(N / 16, M / 16, 1);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // warmup
    wmma_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    wmma_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms;
}

int main(int argc, char** argv)
{
    // 기본 사이즈: 1024 (16의 배수)
    int M = 1024, N = 1024, K = 1024;
    if (argc == 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }

    if (M % 16 || N % 16 || K % 16) {
        printf("M, N, K must be multiples of 16 for WMMA.\n");
        return 1;
    }

    printf("GEMM size: M=%d, N=%d, K=%d\n", M, N, K);

    size_t bytesA = sizeof(float) * M * K;
    size_t bytesB = sizeof(float) * K * N;
    size_t bytesC = sizeof(float) * M * N;

    float *hA = (float*)std::malloc(bytesA);
    float *hB = (float*)std::malloc(bytesB);

    std::srand(0);
    init_host_float(hA, M * K);
    init_host_float(hB, K * N);

    float *dA_f32, *dB_f32, *dC_f32;
    half  *dA_f16, *dB_f16;
    float *dC_tc;

    CHECK_CUDA(cudaMalloc(&dA_f32, bytesA));
    CHECK_CUDA(cudaMalloc(&dB_f32, bytesB));
    CHECK_CUDA(cudaMalloc(&dC_f32, bytesC));

    CHECK_CUDA(cudaMalloc(&dA_f16, sizeof(half) * M * K));
    CHECK_CUDA(cudaMalloc(&dB_f16, sizeof(half) * K * N));
    CHECK_CUDA(cudaMalloc(&dC_tc,  bytesC));

    CHECK_CUDA(cudaMemcpy(dA_f32, hA, bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_f32, hB, bytesB, cudaMemcpyHostToDevice));

    // host에서 half 버퍼 만들고 전송
    half *hA_f16 = (half*)std::malloc(sizeof(half) * M * K);
    half *hB_f16 = (half*)std::malloc(sizeof(half) * K * N);
    float_to_half(hA, hA_f16, M * K);
    float_to_half(hB, hB_f16, K * N);

    CHECK_CUDA(cudaMemcpy(dA_f16, hA_f16, sizeof(half) * M * K, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_f16, hB_f16, sizeof(half) * K * N, cudaMemcpyHostToDevice));

    // --------------------
    // FP32 GEMM
    // --------------------
    float ms_fp32 = run_fp32_gemm(M, N, K, dA_f32, dB_f32, dC_f32);
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops_fp32 = flops / (ms_fp32 * 1.0e6); // 1e9 / 1e3

    printf("[FP32 GEMM]\n");
    printf("  time    = %.3f ms\n", ms_fp32);
    printf("  GFLOPs  = %.2f\n", gflops_fp32);

    // --------------------
    // Tensor Core GEMM (WMMA)
    // --------------------
    float ms_tc = run_wmma_gemm(M, N, K, dA_f16, dB_f16, dC_tc);
    double gflops_tc = flops / (ms_tc * 1.0e6);

    printf("[Tensor Core GEMM (WMMA)]\n");
    printf("  time    = %.3f ms\n", ms_tc);
    printf("  GFLOPs  = %.2f\n", gflops_tc);

    printf("Speedup (FP32 / TC) = %.2fx\n", ms_fp32 / ms_tc);

    // clean up
    std::free(hA);
    std::free(hB);
    std::free(hA_f16);
    std::free(hB_f16);

    CHECK_CUDA(cudaFree(dA_f32));
    CHECK_CUDA(cudaFree(dB_f32));
    CHECK_CUDA(cudaFree(dC_f32));
    CHECK_CUDA(cudaFree(dA_f16));
    CHECK_CUDA(cudaFree(dB_f16));
    CHECK_CUDA(cudaFree(dC_tc));

    return 0;
}
/*
nvcc -O3 -std=c++17 -arch=sm_86 -lineinfo   -o test_tc_gemm test_tc_gemm.cu

# FP32 GEMM만

ncu --kernel-name regex:.*fp32.*gemm.*kernel.*     --set full     --launch-skip 0 --launch-count 1     .\test_tc_gemm.exe
ncu --kernel-name regex:.*wmma.*gemm.*kernel.*    --set full     --launch-skip 0 --launch-count 1     .\test_tc_gemm.exe


*/