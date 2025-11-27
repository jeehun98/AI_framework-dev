// tensorcore_wmma_vs_fp32.cu
// 1.4.1 Test — Tensor Core (WMMA) vs FP32 FMA

#include <cstdio>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

// 행렬 크기 (16의 배수)
constexpr int M = 256;
constexpr int N = 256;
constexpr int K = 256;

// 반복 횟수 (시간 늘리기용)
constexpr int ITERS = 10;

// -----------------------------
// FP32 naive GEMM kernel
// C = A * B
// A: [M x K], B: [K x N], C: [M x N]
// -----------------------------
__global__
void fp32_gemm_kernel(const float* __restrict__ A,
                      const float* __restrict__ B,
                      float* __restrict__ C,
                      int M, int N, int K, int iters)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 0..M-1
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 0..N-1

    if (row >= M || col >= N) return;

    // 여러 번 반복해서 연산량 증가
    for (int it = 0; it < iters; ++it) {
        float acc = 0.0f;
        // 단순 i-k 루프
        for (int k = 0; k < K; ++k) {
            float a = A[row * K + k];
            float b = B[k * N + col];
            acc += a * b;
        }
        // 마지막 it 기준으로 덮어쓰기 (결과 자체는 크게 중요하지 않음)
        C[row * N + col] = acc;
    }
}

// -----------------------------
// WMMA Tensor Core GEMM kernel
// A_half: [M x K] (half)
// B_half: [K x N] (half)
// C:      [M x N] (float, accumulator)
//  - warp당 16x16 tile 하나 계산
//  - row-major / row-major 사용
// -----------------------------
__global__
void wmma_gemm_kernel(const half* __restrict__ A_half,
                      const half* __restrict__ B_half,
                      float* __restrict__ C,
                      int M, int N, int K, int iters)
{
#if __CUDA_ARCH__ >= 700
    // warp 단위 tile
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    // warp 하나가 16x16 tile 하나 담당
    int tile_row = blockIdx.y; // 0..(M/16-1)
    int tile_col = blockIdx.x; // 0..(N/16-1)

    // blockDim.x는 32로 가정 (warp 1개)
    if (tile_row * WMMA_M >= M || tile_col * WMMA_N >= N)
        return;

    // 반복 횟수만큼 GEMM 수행
    for (int it = 0; it < iters; ++it) {
        // C fragment 0으로 초기화
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                       half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                       half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                       float> c_frag;

        wmma::fill_fragment(c_frag, 0.0f);

        // K dimension을 16씩 잘라서 누적
        for (int kk = 0; kk < K; kk += WMMA_K) {
            const half* tile_ptr_A = A_half + (tile_row * WMMA_M) * K + kk;
            const half* tile_ptr_B = B_half + kk * N + (tile_col * WMMA_N);

            // row-major / row-major로 load
            wmma::load_matrix_sync(a_frag, tile_ptr_A, K);
            wmma::load_matrix_sync(b_frag, tile_ptr_B, N);

            // C += A * B
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // C에 store
        float* tile_ptr_C = C + (tile_row * WMMA_M) * N + (tile_col * WMMA_N);
        wmma::store_matrix_sync(tile_ptr_C, c_frag, N, wmma::mem_row_major);
    }
#else
    // Tensor Core 없는 아키텍처에서 컴파일될 때 대비용
    (void)A_half; (void)B_half; (void)C;
    (void)M; (void)N; (void)K; (void)iters;
#endif
}

// -----------------------------
// float -> half 변환 커널
// -----------------------------
__global__
void to_half_kernel(const float* __restrict__ src,
                    half* __restrict__ dst,
                    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

// -----------------------------
// 유틸: CUDA 에러 체크
// -----------------------------
inline void checkCuda(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA Error (%s): %s\n", msg, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

int main()
{
    std::printf("== 1.4.1 Test — Tensor Core (WMMA) vs FP32 FMA ==\n");

    const size_t bytes_A = M * K * sizeof(float);
    const size_t bytes_B = K * N * sizeof(float);
    const size_t bytes_C = M * N * sizeof(float);

    // Host 메모리
    float* h_A = (float*)std::malloc(bytes_A);
    float* h_B = (float*)std::malloc(bytes_B);
    float* h_C = (float*)std::malloc(bytes_C);

    // 간단한 초기화
    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;      // 그냥 1
    for (int i = 0; i < K * N; ++i) h_B[i] = 1.0f;      // 그냥 1

    // Device 메모리
    float *d_A_f32, *d_B_f32, *d_C_fp32, *d_C_wmma;
    half  *d_A_half, *d_B_half;

    checkCuda(cudaMalloc(&d_A_f32, bytes_A), "malloc d_A_f32");
    checkCuda(cudaMalloc(&d_B_f32, bytes_B), "malloc d_B_f32");
    checkCuda(cudaMalloc(&d_C_fp32, bytes_C), "malloc d_C_fp32");
    checkCuda(cudaMalloc(&d_C_wmma, bytes_C), "malloc d_C_wmma");

    checkCuda(cudaMemcpy(d_A_f32, h_A, bytes_A, cudaMemcpyHostToDevice), "memcpy A");
    checkCuda(cudaMemcpy(d_B_f32, h_B, bytes_B, cudaMemcpyHostToDevice), "memcpy B");

    // FP16용 버퍼
    checkCuda(cudaMalloc(&d_A_half, M * K * sizeof(half)), "malloc d_A_half");
    checkCuda(cudaMalloc(&d_B_half, K * N * sizeof(half)), "malloc d_B_half");

    // float → half 변환
    int nA = M * K;
    int nB = K * N;
    dim3 blk(256);
    dim3 grdA((nA + blk.x - 1) / blk.x);
    dim3 grdB((nB + blk.x - 1) / blk.x);

    to_half_kernel<<<grdA, blk>>>(d_A_f32, d_A_half, nA);
    to_half_kernel<<<grdB, blk>>>(d_B_f32, d_B_half, nB);
    checkCuda(cudaGetLastError(), "launch to_half_kernel");
    checkCuda(cudaDeviceSynchronize(), "to_half sync");

    // -----------------------------
    // FP32 naive GEMM 타이밍
    // -----------------------------
    {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x,
                  (M + block.y - 1) / block.y);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        checkCuda(cudaMemset(d_C_fp32, 0, bytes_C), "memset C fp32");

        cudaEventRecord(start);
        fp32_gemm_kernel<<<grid, block>>>(d_A_f32, d_B_f32, d_C_fp32,
                                          M, N, K, ITERS);
        cudaEventRecord(stop);

        checkCuda(cudaGetLastError(), "launch fp32_gemm_kernel");
        checkCuda(cudaDeviceSynchronize(), "sync fp32_gemm_kernel");

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        // FLOPs: 2 * M * N * K * ITERS
        double flops = 2.0 * M * N * K * ITERS;
        double tflops = flops / (ms * 1e-3) / 1e12;

        std::printf("\n[FP32 GEMM]\n");
        std::printf("  Time   = %.3f ms\n", ms);
        std::printf("  TFLOPS = %.3f\n", tflops);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // -----------------------------
    // WMMA Tensor Core GEMM 타이밍
    // -----------------------------
    {
        dim3 block(32, 1, 1); // warp 1개
        dim3 grid(N / 16, M / 16, 1); // warp tile 단위

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        checkCuda(cudaMemset(d_C_wmma, 0, bytes_C), "memset C wmma");

        cudaEventRecord(start);
        wmma_gemm_kernel<<<grid, block>>>(d_A_half, d_B_half, d_C_wmma,
                                          M, N, K, ITERS);
        cudaEventRecord(stop);

        checkCuda(cudaGetLastError(), "launch wmma_gemm_kernel");
        checkCuda(cudaDeviceSynchronize(), "sync wmma_gemm_kernel");

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        double flops = 2.0 * M * N * K * ITERS;
        double tflops = flops / (ms * 1e-3) / 1e12;

        std::printf("\n[WMMA Tensor Core GEMM]\n");
        std::printf("  Time   = %.3f ms\n", ms);
        std::printf("  TFLOPS = %.3f\n", tflops);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // 결과 sanity check (C[0] 정도만)
    checkCuda(cudaMemcpy(h_C, d_C_fp32, bytes_C, cudaMemcpyDeviceToHost), "copy C fp32");
    std::printf("\nSample C[0] (FP32)  = %f\n", h_C[0]);

    checkCuda(cudaMemcpy(h_C, d_C_wmma, bytes_C, cudaMemcpyDeviceToHost), "copy C wmma");
    std::printf("Sample C[0] (WMMA)  = %f\n", h_C[0]);

    // 정리
    cudaFree(d_A_f32);
    cudaFree(d_B_f32);
    cudaFree(d_C_fp32);
    cudaFree(d_C_wmma);
    cudaFree(d_A_half);
    cudaFree(d_B_half);

    std::free(h_A);
    std::free(h_B);
    std::free(h_C);

    return 0;
}

/*
빌드 예시:

nvcc -O3 -arch=sm_86 -lineinfo tensorcore_wmma_vs_fp32.cu -o tensorcore_wmma_vs_fp32.exe

./tensorcore_wmma_vs_fp32.exe

ncu --set full --kernel-name regex:.*fp32_gemm_kernel.* ./tensorcore_wmma_vs_fp32.exe

ncu --set full --kernel-name regex:.*wmma_gemm_kernel.* ./tensorcore_wmma_vs_fp32.exe

*/
