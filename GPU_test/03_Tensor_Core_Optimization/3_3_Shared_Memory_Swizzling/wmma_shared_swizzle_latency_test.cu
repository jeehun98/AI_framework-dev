// wmma_shared_swizzle_latency_test.cu
// nvcc -arch=sm_86 -O3 wmma_shared_swizzle_latency_test.cu -o wmma_shared_swizzle_latency_test.exe

#include <cstdio>
#include <vector>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error %s:%d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

constexpr int M = 128;
constexpr int N = 128;
constexpr int K = 128;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// --- 공통: 간단한 XOR 기반 swizzle 함수 ---
// row, col (16x16 타일 기준) → swizzled column index
__device__ __forceinline__
int swizzle_col_xor(int row, int col, int logical_cols) {
    // row의 하위 비트로 col 하위 비트를 XOR
    // (실제 h/w 최적 swizzle과는 다를 수 있지만, bank pattern 변화를 보기 위한 테스트용)
    int row_group = row & 0x7;            // 0~7
    int col_low   = col & 0x7;            // 0~7
    int col_high  = col & ~0x7;           // 상위 비트
    int swz_low   = col_low ^ row_group;  // 간단 XOR
    int swz       = col_high + (swz_low % logical_cols);
    return swz;
}

// =============================================
// 1. No-swizzle kernel
// =============================================
__global__ void wmma_shared_noswizzle_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    extern __shared__ __half shmem[];

    // A, B 16x16 타일을 위한 shared 영역 (row-major)
    __half* shA = shmem;                        // 16x16
    __half* shB = shmem + WMMA_M * WMMA_K;      // 16x16

    int lane_id   = threadIdx.x;                // 0..31 (warp 단위)
    int warp_row  = blockIdx.y;                 // 0..(M/16-1)
    int warp_col  = blockIdx.x;                 // 0..(N/16-1)

    // 이 warp가 담당하는 C 타일의 시작 위치
    int c_row = warp_row * WMMA_M;
    int c_col = warp_col * WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                   float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // K dimension을 16씩 쪼개서 반복 (latency/stall 관찰용으로 일부 반복)
    for (int ko = 0; ko < K; ko += WMMA_K) {

        // --- global → shared (no swizzle, 순수 row-major) ---
        // 16x16 타일을 warp 32 thread가 협업해서 로드
        for (int idx = lane_id; idx < WMMA_M * WMMA_K; idx += 32) {
            int r = idx / WMMA_K;
            int c = idx % WMMA_K;

            int gA_r = c_row + r;
            int gA_c = ko     + c;
            int gB_r = ko     + r;
            int gB_c = c_col + c;

            shA[r * WMMA_K + c] = A[gA_r * K + gA_c]; // row-major
            shB[r * WMMA_N + c] = B[gB_r * N + gB_c]; // col-major로 읽지만 row-major로 저장 후 load_matrix_sync에서 col-major 해석
        }

        __syncthreads();

        // --- shared → fragment (ldmatrix 내부 사용) ---
        wmma::load_matrix_sync(a_frag, shA, WMMA_K);
        // B는 col_major fragment이므로 leading dimension = WMMA_N
        wmma::load_matrix_sync(b_frag, shB, WMMA_N);

        // --- MMA ---
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // --- C store ---
    float* C_tile = C + c_row * N + c_col;
    wmma::store_matrix_sync(C_tile, c_frag, N, wmma::mem_row_major);
}

// =============================================
// 2. Swizzle kernel (XOR-based shared layout)
// =============================================
__global__ void wmma_shared_swizzle_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    extern __shared__ __half shmem[];

    // padding + swizzle를 위해 stride를 넉넉히 준 2D 레이아웃
    // 논리적으로는 16x16이지만, shared 상 stride를 늘려 bank 패턴을 바꿈
    constexpr int STRIDE = WMMA_K + 8; // 16 + 8 padding
    __half* shA = shmem;                       // 16 x STRIDE
    __half* shB = shmem + WMMA_M * STRIDE;     // 16 x STRIDE

    int lane_id   = threadIdx.x;
    int warp_row  = blockIdx.y;
    int warp_col  = blockIdx.x;

    int c_row = warp_row * WMMA_M;
    int c_col = warp_col * WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                   float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int ko = 0; ko < K; ko += WMMA_K) {

        // --- global → shared (swizzled layout) ---
        for (int idx = lane_id; idx < WMMA_M * WMMA_K; idx += 32) {
            int r = idx / WMMA_K;
            int c = idx % WMMA_K;

            int gA_r = c_row + r;
            int gA_c = ko     + c;
            int gB_r = ko     + r;
            int gB_c = c_col + c;

            int sc = swizzle_col_xor(r, c, WMMA_K); // 0~15 범위에서 XOR만 적용
            int sA_idx = r * STRIDE + sc;
            int sB_idx = r * STRIDE + sc;

            shA[sA_idx] = A[gA_r * K + gA_c];
            shB[sB_idx] = B[gB_r * N + gB_c];
        }

        __syncthreads();

        // --- shared → fragment ---
        // load_matrix_sync 에 stride=STRIDE 전달 → swizzled row-major에서 ldmatrix load
        wmma::load_matrix_sync(a_frag, shA, STRIDE);
        wmma::load_matrix_sync(b_frag, shB, STRIDE);

        // --- MMA ---
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // --- C store ---
    float* C_tile = C + c_row * N + c_col;
    wmma::store_matrix_sync(C_tile, c_frag, N, wmma::mem_row_major);
}

// =============================================
// Host-side: 타이밍 측정 및 실행
// =============================================
int main() {
    printf("WMMA shared swizzle latency test (M=%d, N=%d, K=%d)\n", M, N, K);

    size_t bytesA = M * K * sizeof(__half);
    size_t bytesB = K * N * sizeof(__half);
    size_t bytesC = M * N * sizeof(float);

    __half* dA;
    __half* dB;
    float*  dC_noswizzle;
    float*  dC_swizzle;

    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC_noswizzle, bytesC));
    CHECK_CUDA(cudaMalloc(&dC_swizzle, bytesC));

    // Host 측에서 A, B 를 모두 1.0으로 초기화 (MMA throughput 확인용)
    std::vector<__half> hA(M * K), hB(K * N);
    for (int i = 0; i < M * K; ++i) {
        hA[i] = __float2half(1.0f);
    }
    for (int i = 0; i < K * N; ++i) {
        hB[i] = __float2half(1.0f);
    }

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

    dim3 blockDim(32, 1, 1);                    // warp 1개
    dim3 gridDim(N / WMMA_N, M / WMMA_M, 1);    // 8x8 = 64 warps

    // shared memory size 설정
    size_t shmemBytesNoswizzle = (WMMA_M * WMMA_K      // A 16x16
                                + WMMA_K * WMMA_N)     // B 16x16
                                * sizeof(__half);

    size_t STRIDE = WMMA_K + 8;
    size_t shmemBytesSwizzle  = (WMMA_M * STRIDE       // A 16 x STRIDE
                               + WMMA_M * STRIDE)      // B 16 x STRIDE
                               * sizeof(__half);

    // --- warm-up ---
    wmma_shared_noswizzle_kernel<<<gridDim, blockDim, shmemBytesNoswizzle>>>(
        dA, dB, dC_noswizzle, M, N, K);
    wmma_shared_swizzle_kernel<<<gridDim, blockDim, shmemBytesSwizzle>>>(
        dA, dB, dC_swizzle, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // --- timing 설정 ---
    const int NUM_ITER = 100;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 1) no-swizzle
    CHECK_CUDA(cudaEventRecord(start));
    for (int it = 0; it < NUM_ITER; ++it) {
        wmma_shared_noswizzle_kernel<<<gridDim, blockDim, shmemBytesNoswizzle>>>(
            dA, dB, dC_noswizzle, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_noswizzle = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_noswizzle, start, stop));
    ms_noswizzle /= NUM_ITER;

    // 2) swizzle
    CHECK_CUDA(cudaEventRecord(start));
    for (int it = 0; it < NUM_ITER; ++it) {
        wmma_shared_swizzle_kernel<<<gridDim, blockDim, shmemBytesSwizzle>>>(
            dA, dB, dC_swizzle, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_swizzle = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_swizzle, start, stop));
    ms_swizzle /= NUM_ITER;

    printf("[Timing] no-swizzle : %.3f ms\n", ms_noswizzle);
    printf("[Timing] swizzle    : %.3f ms\n", ms_swizzle);

    // 간단 검증: C 값이 제대로 누적됐는지 (원하면 추가)
    // expected = K (128) * 1.0 * 1.0 = 128.0
    std::vector<float> hC(M * N);
    CHECK_CUDA(cudaMemcpy(hC.data(), dC_noswizzle, bytesC, cudaMemcpyDeviceToHost));
    float minv = 1e9f, maxv = -1e9f;
    for (int i = 0; i < M * N; ++i) {
        if (hC[i] < minv) minv = hC[i];
        if (hC[i] > maxv) maxv = hC[i];
    }
    printf("[Check] C_noswizzle value range: min=%.1f, max=%.1f (expected around %.1f)\n",
           minv, maxv, float(K));

    // 정리
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC_noswizzle));
    CHECK_CUDA(cudaFree(dC_swizzle));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}


/*
nvcc -std=c++17 -O3   -arch=sm_86   wmma_shared_swizzle_latency_test.cu   -o wmma_shared_swizzle_latency_test.exe

ncu   --kernel-name regex:.*wmma_shared_noswizzle_kernel.*   --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum,smsp__warp_issue_stalled_memory_throttle_per_warp_active.avg   --set full --launch-skip 0 --launch-count 1  .\wmma_shared_swizzle_latency_test.exe

ncu   --kernel-name regex:.*wmma_shared_swizzle_kernel.*   --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum,smsp__warp_issue_stalled_memory_throttle_per_warp_active.avg   --set full --launch-skip 0 --launch-count 1  .\wmma_shared_swizzle_latency_test.exe

ncu ^
  --kernel-name regex:.*wmma_shared_swizzle_kernel.* ^
  --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum,smsp__warp_issue_stalled_memory_throttle_per_warp_active.avg ^
  --set full ^
  .\wmma_shared_swizzle_latency_test.exe

*/