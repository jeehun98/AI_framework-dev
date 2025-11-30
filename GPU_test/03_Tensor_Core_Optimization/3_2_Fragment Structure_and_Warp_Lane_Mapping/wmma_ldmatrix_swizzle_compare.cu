// wmma_ldmatrix_swizzle_compare.cu
#include <cstdio>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err__));                                \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
#endif

// WMMA tile shape
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// 전체 GEMM 크기 (tile 배수)
constexpr int M = 128;
constexpr int N = 128;
constexpr int K = 128;

// 반복 횟수: shared load/stall을 좀 키우기 위한 loop
constexpr int K_TILE_ITERS = K / WMMA_K;
constexpr int MMA_REPEATS  = 16;   // 같은 tile에서 여러 번 MMA 수행해서 tensor pipe 충분히 사용

// 간단 초기화: A = i % 3, B = j % 5
__global__ void init_matrix_half(half* A, int rows, int cols, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = rows * cols;
    if (idx >= size) return;

    int r = idx / cols;
    int c = idx % cols;
    float val = scale * ((r + c) % 7);
    A[idx] = __float2half(val);
}

// ===== swizzle 유틸 (예시용) =====

// 간단 XOR-based swizzle (예시용)
// - row는 그대로 두고, col의 하위 몇 비트에 row의 일부 비트를 XOR
// - 실제 conflict=0 보장을 하진 않지만, bank 패턴을 “확실히 다르게” 만드는 용도
__device__ __forceinline__ int swizzle_index_row_major(int row, int col, int ld) {
    // ld = leading dimension (K 또는 N)
    // row를 0~15, col을 0~15로 가정
    int row_group = (row & 0x3);          // 4-row 그룹
    int col_swizzled = col ^ (row_group * 4);
    col_swizzled &= 0xF;                  // 0~15 안으로
    return row * ld + col_swizzled;
}

// ======= 커널 1: shared linear 배치 (no swizzle) =======

__global__ void wmma_ldmatrix_noswizzle_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    extern __shared__ half smem[];
    half* As = smem;                               // [WMMA_M x WMMA_K] row-major
    half* Bs = smem + WMMA_M * WMMA_K;             // [WMMA_K x WMMA_N] col-major

    // block당 1 warp만 사용
    int lane_id = threadIdx.x & 31;
    if (threadIdx.x >= 32) return;

    // C tile 위치
    int tile_m = blockIdx.y;   // 0..(M/16-1)
    int tile_n = blockIdx.x;   // 0..(N/16-1)

    int c_row = tile_m * WMMA_M;
    int c_col = tile_n * WMMA_N;

    // accumulator fragment
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // 반복해서 Tensor Core 파이프를 충분히 사용
    for (int rep = 0; rep < MMA_REPEATS; ++rep) {
        for (int kt = 0; kt < K_TILE_ITERS; ++kt) {
            int a_row_base = c_row;
            int a_col_base = kt * WMMA_K;

            int b_row_base = kt * WMMA_K;
            int b_col_base = c_col;

            // ===== Global -> Shared Load (no swizzle) =====
            // As: row-major (M x K), tile 16x16
            // 256 elements / 32 threads = 8 elements per thread
            for (int i = 0; i < (WMMA_M * WMMA_K) / 32; ++i) {
                int idx = lane_id + i * 32;
                int r = idx / WMMA_K;
                int c = idx % WMMA_K;

                int gr = a_row_base + r;
                int gc = a_col_base + c;
                if (gr < M && gc < K)
                    As[r * WMMA_K + c] = A[gr * K + gc];
                else
                    As[r * WMMA_K + c] = __float2half(0.0f);
            }

            // Bs: col-major (K x N)
            for (int i = 0; i < (WMMA_K * WMMA_N) / 32; ++i) {
                int idx = lane_id + i * 32;
                int r = idx / WMMA_N;   // 0..15 (K dimension)
                int c = idx % WMMA_N;   // 0..15 (N dimension)

                int gr = b_row_base + r;
                int gc = b_col_base + c;
                if (gr < K && gc < N)
                    // col-major: row=r, col=c -> offset = r + c*ld
                    Bs[r + c * WMMA_K] = B[gr + gc * K];
                else
                    Bs[r + c * WMMA_K] = __float2half(0.0f);
            }

            __syncthreads();

            // ===== Shared -> WMMA fragment (ldmatrix 경로) =====
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

            wmma::load_matrix_sync(a_frag, As, WMMA_K);
            wmma::load_matrix_sync(b_frag, Bs, WMMA_K);

            // MMA
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

            __syncthreads();
        }
    }

    // C에 저장 (row-major)
    // 여기서는 단순히 1 번만 저장 (rep 누적 포함)
    float* C_tile = C + c_row * N + c_col;
    wmma::store_matrix_sync(C_tile, c_frag, N, wmma::mem_row_major);
}

// ======= 커널 2: shared swizzle 배치 (예시 XOR swizzle) =======

__global__ void wmma_ldmatrix_swizzle_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    extern __shared__ half smem[];
    half* As = smem;                               // swizzled row-major
    half* Bs = smem + WMMA_M * WMMA_K;             // swizzled "col-major-ish"

    int lane_id = threadIdx.x & 31;
    if (threadIdx.x >= 32) return;

    int tile_m = blockIdx.y;
    int tile_n = blockIdx.x;

    int c_row = tile_m * WMMA_M;
    int c_col = tile_n * WMMA_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int rep = 0; rep < MMA_REPEATS; ++rep) {
        for (int kt = 0; kt < K_TILE_ITERS; ++kt) {
            int a_row_base = c_row;
            int a_col_base = kt * WMMA_K;

            int b_row_base = kt * WMMA_K;
            int b_col_base = c_col;

            // ===== Global -> Shared Load (swizzled) =====
            // A tile swizzle
            for (int i = 0; i < (WMMA_M * WMMA_K) / 32; ++i) {
                int idx = lane_id + i * 32;
                int r = idx / WMMA_K;
                int c = idx % WMMA_K;

                int gr = a_row_base + r;
                int gc = a_col_base + c;
                half val = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);

                int sidx = swizzle_index_row_major(r, c, WMMA_K);
                As[sidx] = val;
            }

            // B tile swizzle
            // 원래는 col-major 형태지만, 여기서는 row-major 축을 기준으로 swizzle 적용 (실험용)
            for (int i = 0; i < (WMMA_K * WMMA_N) / 32; ++i) {
                int idx = lane_id + i * 32;
                int r = idx / WMMA_N;   // 0..15 (K)
                int c = idx % WMMA_N;   // 0..15 (N)

                int gr = b_row_base + r;
                int gc = b_col_base + c;
                half val = (gr < K && gc < N) ? B[gr + gc * K] : __float2half(0.0f);

                int sidx = swizzle_index_row_major(r, c, WMMA_N);
                Bs[sidx] = val;
            }

            __syncthreads();

            // ===== Shared -> WMMA fragment =====
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

            // NOTE: 여기서는 As/Bs를 여전히 "정상 ld"로 읽는다고 가정하고 load.
            // 실제로는 swizzle layout과 wmma::load_matrix_sync 레이아웃이 안 맞으면
            // GEMM 결과는 엉망이지만, shared load 패턴/충돌을 보는 데는 문제 없음.
            wmma::load_matrix_sync(a_frag, As, WMMA_K);
            wmma::load_matrix_sync(b_frag, Bs, WMMA_K);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

            __syncthreads();
        }
    }

    float* C_tile = C + c_row * N + c_col;
    wmma::store_matrix_sync(C_tile, c_frag, N, wmma::mem_row_major);
}

// ================== host 테스트 코드 ==================

int main() {
    printf("WMMA ldmatrix swizzle vs no-swizzle test (M=%d, N=%d, K=%d)\n", M, N, K);

    size_t bytesA = sizeof(half) * M * K;
    size_t bytesB = sizeof(half) * K * N;
    size_t bytesC = sizeof(float) * M * N;

    half* dA = nullptr;
    half* dB = nullptr;
    float* dC_noswizzle = nullptr;
    float* dC_swizzle   = nullptr;

    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC_noswizzle, bytesC));
    CHECK_CUDA(cudaMalloc(&dC_swizzle, bytesC));

    // 초기화
    {
        int threads = 256;
        int blocksA = (M * K + threads - 1) / threads;
        int blocksB = (K * N + threads - 1) / threads;
        init_matrix_half<<<blocksA, threads>>>(dA, M, K, 0.1f);
        init_matrix_half<<<blocksB, threads>>>(dB, K, N, 0.2f);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    dim3 blockDim(32, 1, 1);                  // 1 warp
    dim3 gridDim(N / WMMA_N, M / WMMA_M, 1);  // C tiles

    size_t shmem_bytes = (WMMA_M * WMMA_K + WMMA_K * WMMA_N) * sizeof(half);

    // warmup
    wmma_ldmatrix_noswizzle_kernel<<<gridDim, blockDim, shmem_bytes>>>(dA, dB, dC_noswizzle, M, N, K);
    wmma_ldmatrix_swizzle_kernel<<<gridDim, blockDim, shmem_bytes>>>(dA, dB, dC_swizzle, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // no swizzle
    CHECK_CUDA(cudaEventRecord(start));
    wmma_ldmatrix_noswizzle_kernel<<<gridDim, blockDim, shmem_bytes>>>(dA, dB, dC_noswizzle, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_noswizzle = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_noswizzle, start, stop));

    // swizzle
    CHECK_CUDA(cudaEventRecord(start));
    wmma_ldmatrix_swizzle_kernel<<<gridDim, blockDim, shmem_bytes>>>(dA, dB, dC_swizzle, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_swizzle = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_swizzle, start, stop));

    printf("[Timing] no-swizzle : %.3f ms\n", ms_noswizzle);
    printf("[Timing] swizzle    : %.3f ms\n", ms_swizzle);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC_noswizzle));
    CHECK_CUDA(cudaFree(dC_swizzle));

    return 0;
}
/*
nvcc -std=c++17 -O3   -arch=sm_86   wmma_ldmatrix_swizzle_compare.cu   -o wmma_ldmatrix_swizzle_compare.exe

ncu   --kernel-name regex:.*wmma_ldmatrix_noswizzle_kernel.*   --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum,smsp__warp_issue_stalled_lg_throttle_per_warp_active.avg,smsp__warp_issue_stalled_membar_per_warp_active.avg,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.avg   --set full   --launch-skip 0 --launch-count 1   .\wmma_ldmatrix_swizzle_compare.exe
ncu   --kernel-name regex:.*wmma_ldmatrix_swizzle_kernel.*   --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum,smsp__warp_issue_stalled_lg_throttle_per_warp_active.avg,smsp__warp_issue_stalled_membar_per_warp_active.avg,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.avg   --set full   --launch-skip 0 --launch-count 1   .\wmma_ldmatrix_swizzle_compare.exe
*/