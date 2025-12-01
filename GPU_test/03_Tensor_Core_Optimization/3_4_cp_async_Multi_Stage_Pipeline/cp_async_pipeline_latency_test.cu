// cp_async_pipeline_latency_test.cu
// nvcc -arch=sm_86 -O3 -lineinfo cp_async_pipeline_latency_test.cu -o cp_async_pipeline_latency_test.exe

#include <cstdio>
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>

using namespace nvcuda;

#define CUDA_CHECK(err)                                      \
    do {                                                     \
        cudaError_t e = (err);                               \
        if (e != cudaSuccess) {                              \
            printf("CUDA Error %s:%d: %s\n",                 \
                   __FILE__, __LINE__, cudaGetErrorString(e));\
            exit(1);                                         \
        }                                                    \
    } while (0)

constexpr int M = 128;
constexpr int N = 128;
constexpr int K = 128;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// -------------------- cp.async wrapper --------------------

template<int BYTES>
__device__ inline void cp_async_ca_shared_global(void* smem_ptr, const void* gmem_ptr) {
#if __CUDA_ARCH__ >= 800
    // shared pointer must be converted to 32-bit shared address
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));

    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], %2;\n" ::
        "r"(smem_addr), "l"(gmem_ptr), "n"(BYTES)
    );
#endif
}

__device__ inline void cp_async_commit_group() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ inline void cp_async_wait_group_0() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

// -------------------- Naive kernel: ld.global → smem → MMA --------------------

__global__ void wmma_cp_async_naive_kernel(const half* __restrict__ A,
                                           const half* __restrict__ B,
                                           float* __restrict__ C,
                                           int M, int N, int K)
{
    extern __shared__ half smem[];
    // A, B tile (each 16x16), single stage
    half* smemA = smem;
    half* smemB = smem + WMMA_M * WMMA_K;

    int lane = threadIdx.x;  // 0..31
    int block_row = blockIdx.y; // tile row index
    int block_col = blockIdx.x; // tile col index

    // one warp computes one 16x16 tile of C
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // A: row-major, B: col-major
    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        // ---- A tile load (row-major) ----
        // 16x16 -> 16 rows × 2 segments per row (8 elements) = 32 segments
        int a_row = lane / 2;       // 0..15
        int a_seg = lane % 2;       // 0 or 1
        int a_col_start = a_seg * 8;

        int global_a_row = block_row * WMMA_M + a_row;
        const half* gmemA = A + global_a_row * K + (k0 + a_col_start);
        half* smemA_row = smemA + a_row * WMMA_K + a_col_start;

        for (int i = 0; i < 8; ++i) {
            smemA_row[i] = gmemA[i];
        }

        // ---- B tile load (col-major) ----
        // 16x16 -> 16 cols × 2 segments per col (8 elements in K direction)
        int b_col = lane / 2;       // 0..15
        int b_seg = lane % 2;       // 0 or 1
        int b_row_start = b_seg * 8;

        int global_b_col = block_col * WMMA_N + b_col;
        const half* gmemB = B + (k0 + b_row_start) + global_b_col * K;
        half* smemB_col = smemB + b_row_start + b_col * WMMA_K;

        for (int i = 0; i < 8; ++i) {
            smemB_col[i] = gmemB[i];  // col-major in smem: row + col*lda
        }

        __syncthreads();

        // ---- MMA ----
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

        wmma::load_matrix_sync(a_frag, smemA, WMMA_K);
        wmma::load_matrix_sync(b_frag, smemB, WMMA_K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // ---- store C ----
    int global_c_row = block_row * WMMA_M;
    int global_c_col = block_col * WMMA_N;
    float* C_tile = C + global_c_row * N + global_c_col;

    wmma::store_matrix_sync(C_tile, c_frag, N, wmma::mem_row_major);
}

// -------------------- cp.async 2-stage pipeline kernel --------------------

__global__ void wmma_cp_async_pipeline_kernel(const half* __restrict__ A,
                                              const half* __restrict__ B,
                                              float* __restrict__ C,
                                              int M, int N, int K)
{
    extern __shared__ half smem[];
    // double-buffer: A(2) + B(2)
    half* smemA0 = smem;
    half* smemA1 = smemA0 + WMMA_M * WMMA_K;
    half* smemB0 = smemA1 + WMMA_M * WMMA_K;
    half* smemB1 = smemB0 + WMMA_K * WMMA_N;

    half* smemA[2] = { smemA0, smemA1 };
    half* smemB[2] = { smemB0, smemB1 };

    int lane = threadIdx.x;  // 0..31
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int stage = 0;
    int next_stage = 1;

    // -------------------- Preload k0 = 0 tile with cp.async --------------------

    int k0 = 0;
    {
        // A tile
        int a_row = lane / 2;       // 0..15
        int a_seg = lane % 2;       // 0 or 1
        int a_col_start = a_seg * 8;

        int global_a_row = block_row * WMMA_M + a_row;
        const half* gmemA = A + global_a_row * K + (k0 + a_col_start);
        half* smemA_row = smemA[stage] + a_row * WMMA_K + a_col_start;

        cp_async_ca_shared_global<16>(smemA_row, gmemA);

        // B tile
        int b_col = lane / 2;       // 0..15
        int b_seg = lane % 2;       // 0 or 1
        int b_row_start = b_seg * 8;

        int global_b_col = block_col * WMMA_N + b_col;
        const half* gmemB = B + (k0 + b_row_start) + global_b_col * K;
        half* smemB_col = smemB[stage] + b_row_start + b_col * WMMA_K;

        cp_async_ca_shared_global<16>(smemB_col, gmemB);

        cp_async_commit_group();
        cp_async_wait_group_0();
        __syncthreads();
    }

    // -------------------- K loop with 2-stage pipeline --------------------

    for (int k0_iter = 0; k0_iter < K; k0_iter += WMMA_K) {

        // prefetch next tile if exists
        if (k0_iter + WMMA_K < K) {
            int k1 = k0_iter + WMMA_K;

            // A next tile
            int a_row = lane / 2;
            int a_seg = lane % 2;
            int a_col_start = a_seg * 8;

            int global_a_row = block_row * WMMA_M + a_row;
            const half* gmemA_next = A + global_a_row * K + (k1 + a_col_start);
            half* smemA_row_next = smemA[next_stage] + a_row * WMMA_K + a_col_start;

            cp_async_ca_shared_global<16>(smemA_row_next, gmemA_next);

            // B next tile
            int b_col = lane / 2;
            int b_seg = lane % 2;
            int b_row_start = b_seg * 8;

            int global_b_col = block_col * WMMA_N + b_col;
            const half* gmemB_next = B + (k1 + b_row_start) + global_b_col * K;
            half* smemB_col_next = smemB[next_stage] + b_row_start + b_col * WMMA_K;

            cp_async_ca_shared_global<16>(smemB_col_next, gmemB_next);

            cp_async_commit_group();
        }

        // ---- current tile MMA ----
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

        wmma::load_matrix_sync(a_frag, smemA[stage], WMMA_K);
        wmma::load_matrix_sync(b_frag, smemB[stage], WMMA_K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        // ---- switch stage ----
        if (k0_iter + WMMA_K < K) {
            cp_async_wait_group_0();
            __syncthreads();
            stage ^= 1;
            next_stage ^= 1;
        }
    }

    // -------------------- store C --------------------
    int global_c_row = block_row * WMMA_M;
    int global_c_col = block_col * WMMA_N;
    float* C_tile = C + global_c_row * N + global_c_col;

    wmma::store_matrix_sync(C_tile, c_frag, N, wmma::mem_row_major);
}

// -------------------- Host utility --------------------

void init_matrix_half(half* h, int rows, int cols, float val) {
    for (int i = 0; i < rows * cols; ++i) {
        h[i] = __float2half(val);
    }
}

void check_C_range(const float* hC, int M, int N, const char* name) {
    float mn = hC[0], mx = hC[0];
    for (int i = 1; i < M * N; ++i) {
        mn = std::min(mn, hC[i]);
        mx = std::max(mx, hC[i]);
    }
    printf("[Check] %s value range: min=%f, max=%f (expected around %d.0)\n",
           name, mn, mx, K);
}

int main() {
    printf("WMMA cp.async multi-stage pipeline latency test (M=%d, N=%d, K=%d)\n",
           M, N, K);

    size_t bytes_A = sizeof(half) * M * K;
    size_t bytes_B = sizeof(half) * K * N;
    size_t bytes_C = sizeof(float) * M * N;

    half* hA = (half*)malloc(bytes_A);
    half* hB = (half*)malloc(bytes_B);
    float* hC_naive = (float*)malloc(bytes_C);
    float* hC_pipe  = (float*)malloc(bytes_C);

    init_matrix_half(hA, M, K, 1.0f);
    init_matrix_half(hB, K, N, 1.0f);

    half* dA; half* dB;
    float* dC_naive; float* dC_pipe;

    CUDA_CHECK(cudaMalloc(&dA, bytes_A));
    CUDA_CHECK(cudaMalloc(&dB, bytes_B));
    CUDA_CHECK(cudaMalloc(&dC_naive, bytes_C));
    CUDA_CHECK(cudaMalloc(&dC_pipe, bytes_C));

    CUDA_CHECK(cudaMemcpy(dA, hA, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC_naive, 0, bytes_C));
    CUDA_CHECK(cudaMemset(dC_pipe, 0, bytes_C));

    dim3 grid(N / WMMA_N, M / WMMA_M); // (8,8)
    dim3 block(32);                    // single warp

    // shared: naive = A(16x16) + B(16x16)
    size_t shmem_naive = (WMMA_M * WMMA_K + WMMA_K * WMMA_N) * sizeof(half);
    // shared: pipeline = 2*A + 2*B
    size_t shmem_pipe  = 2 * (WMMA_M * WMMA_K + WMMA_K * WMMA_N) * sizeof(half);

    // warm-up
    wmma_cp_async_naive_kernel<<<grid, block, shmem_naive>>>(dA, dB, dC_naive, M, N, K);
    wmma_cp_async_pipeline_kernel<<<grid, block, shmem_pipe>>>(dA, dB, dC_pipe, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // timing: naive
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMemset(dC_naive, 0, bytes_C));
    CUDA_CHECK(cudaEventRecord(start));
    wmma_cp_async_naive_kernel<<<grid, block, shmem_naive>>>(dA, dB, dC_naive, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_naive = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));

    // timing: pipeline
    CUDA_CHECK(cudaMemset(dC_pipe, 0, bytes_C));
    CUDA_CHECK(cudaEventRecord(start));
    wmma_cp_async_pipeline_kernel<<<grid, block, shmem_pipe>>>(dA, dB, dC_pipe, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_pipe = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_pipe, start, stop));

    printf("[Timing] naive     : %.3f ms\n", ms_naive);
    printf("[Timing] cp.async  : %.3f ms\n", ms_pipe);

    // result check
    CUDA_CHECK(cudaMemcpy(hC_naive, dC_naive, bytes_C, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hC_pipe,  dC_pipe,  bytes_C, cudaMemcpyDeviceToHost));

    check_C_range(hC_naive, M, N, "C_naive");
    check_C_range(hC_pipe,  M, N, "C_cp_async");

    // cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC_naive);
    cudaFree(dC_pipe);
    free(hA);
    free(hB);
    free(hC_naive);
    free(hC_pipe);

    return 0;
}

/*
Example Nsight Compute commands (Linux style):

ncu --kernel-name regex:.*wmma_cp_async_naive_kernel.*     --metrics smsp__pipe_tensor_cycles_active,smsp__warp_issue_stalled_lg_throttle_per_warp_active.avg,dram__bytes_read.sum     --set full --launch-skip 0 --launch-count 1     ./cp_async_pipeline_latency_test.exe

ncu --kernel-name regex:.*wmma_cp_async_pipeline_kernel.*     --metrics smsp__pipe_tensor_cycles_active,smsp__warp_issue_stalled_lg_throttle_per_warp_active.avg,dram__bytes_read.sum     --set full --launch-skip 0 --launch-count 1     ./cp_async_pipeline_latency_test.exe

*/
