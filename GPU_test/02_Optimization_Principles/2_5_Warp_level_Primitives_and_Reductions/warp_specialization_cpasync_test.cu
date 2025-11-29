#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)
#endif

// ------------------------
// cp.async helpers (16B)
// ------------------------
__device__ __forceinline__ void cp_async_16(const void* smem_ptr,
                                            const void* gmem_ptr) {
    // shared memory 주소를 32-bit 주소로 변환
    unsigned int smem_addr =
        static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));

#if !defined(__INTELLISENSE__)
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n" ::
        "r"(smem_addr),   // 32-bit shared addr
        "l"(gmem_ptr)     // 64-bit global addr
    );
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if !defined(__INTELLISENSE__)
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait() {
#if !defined(__INTELLISENSE__)
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

// ------------------------
// GEMM 설정
// ------------------------
constexpr int M = 128;
constexpr int N = 128;
constexpr int K = 128;

// 한 block = C의 16x16 타일 하나
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int TILE_M = WMMA_M;
constexpr int TILE_N = WMMA_N;
constexpr int TILE_K = WMMA_K;

constexpr int BLOCK_ROW_TILES = M / TILE_M;   // 8
constexpr int BLOCK_COL_TILES = N / TILE_N;   // 8

// ------------------------
// monolithic: 단일 warp가 cp.async + MMA 둘 다 수행
// blockDim.x = 32
// ------------------------
__global__ void cpasync_mono_kernel(const half* __restrict__ A,
                                    const half* __restrict__ B,
                                    float* __restrict__ C)
{
    // 단일 warp block 가정
    int lane_id = threadIdx.x & 31;

    // (blockIdx.x, blockIdx.y) 가 C 타일 좌표
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    // C 타일의 시작 좌표
    int c_row = block_row * TILE_M;
    int c_col = block_col * TILE_N;

    extern __shared__ half smem[];
    // double buffering: [2][tileA + tileB]
    half* Asmem = smem;                                      // size: 2 * TILE_M * TILE_K
    half* Bsmem = Asmem + 2 * TILE_M * TILE_K;               // size: 2 * TILE_K * TILE_N

    // WMMA fragment
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const int num_k_tiles = K / TILE_K; // 8
    int buf = 0;

    // 첫 타일 prefetch (K-block 0)
    {
        int elements_per_tile = TILE_M * TILE_K; // 256
        int elems_per_thread = 8;                // 8 half = 16B

        int idx = lane_id * elems_per_thread;    // 0..248

        if (idx < elements_per_tile) {
            int row = idx / TILE_K;
            int col = idx % TILE_K;

            // A tile: (c_row + row, k0 + col)
            int a_row = c_row + row;
            int a_col = 0 + col;

            const half* gA = A + a_row * K + a_col;
            half* sA = Asmem + buf * (TILE_M * TILE_K) + row * TILE_K + col;

            cp_async_16(sA, gA);
        }

        // B 타일은 (k0 + row, c_col + col)
        int elements_per_tile_B = TILE_K * TILE_N; // 256
        if (idx < elements_per_tile_B) {
            int row = idx / TILE_N;
            int col = idx % TILE_N;

            int b_row = 0 + row;
            int b_col = c_col + col;

            const half* gB = B + b_row * N + b_col;
            half* sB = Bsmem + buf * (TILE_K * TILE_N) + row * TILE_N + col;

            cp_async_16(sB, gB);
        }

        cp_async_commit();
        cp_async_wait();
        __syncthreads();
    }

#pragma unroll
    for (int tk = 0; tk < num_k_tiles; ++tk) {
        // 현재 buffer에서 A/B 타일 읽기
        half* sA = Asmem + buf * (TILE_M * TILE_K);
        half* sB = Bsmem + buf * (TILE_K * TILE_N);

        // WMMA용 fragment
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

        wmma::load_matrix_sync(a_frag, sA, TILE_K);
        wmma::load_matrix_sync(b_frag, sB, TILE_N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        int next_k = tk + 1;
        int next_buf = buf ^ 1;

        if (next_k < num_k_tiles) {
            // 다음 타일 prefetch (cp.async)
            int elements_per_tile = TILE_M * TILE_K;
            int elems_per_thread = 8;
            int idx = lane_id * elems_per_thread;

            if (idx < elements_per_tile) {
                int row = idx / TILE_K;
                int col = idx % TILE_K;

                int a_row = c_row + row;
                int a_col = next_k * TILE_K + col;

                const half* gA = A + a_row * K + a_col;
                half* sA2 = Asmem + next_buf * (TILE_M * TILE_K) + row * TILE_K + col;

                cp_async_16(sA2, gA);
            }

            int elements_per_tile_B = TILE_K * TILE_N;
            if (idx < elements_per_tile_B) {
                int row = idx / TILE_N;
                int col = idx % TILE_N;

                int b_row = next_k * TILE_K + row;
                int b_col = c_col + col;

                const half* gB = B + b_row * N + b_col;
                half* sB2 = Bsmem + next_buf * (TILE_K * TILE_N) + row * TILE_N + col;

                cp_async_16(sB2, gB);
            }

            cp_async_commit();
        }

        if (next_k < num_k_tiles) {
            cp_async_wait();
            __syncthreads();
            buf ^= 1;
        }
    }

    // 결과 C에 저장
    // 단일 warp가 16x16 하나만 담당
    float* c_tile_ptr = C + c_row * N + c_col;
    wmma::store_matrix_sync(c_tile_ptr, c_frag, N, wmma::mem_row_major);
}

// ------------------------
// warp specialization 버전
// warp0: cp.async load
// warp1: MMA compute
// blockDim.x = 64 (2 warps)
// ------------------------
__global__ void cpasync_ws_kernel(const half* __restrict__ A,
                                  const half* __restrict__ B,
                                  float* __restrict__ C)
{
    int tid     = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid & 31;

    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    int c_row = block_row * TILE_M;
    int c_col = block_col * TILE_N;

    extern __shared__ half smem[];
    half* Asmem = smem;
    half* Bsmem = Asmem + 2 * TILE_M * TILE_K;

    const int num_k_tiles = K / TILE_K; // 8
    int buf = 0;

    // compute warp의 accumulator
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    if (warp_id == 1) {
        wmma::fill_fragment(c_frag, 0.0f);
    }

    // 첫 타일 prefetch (warp0만 cp.async)
    if (warp_id == 0) {
        int elements_per_tile = TILE_M * TILE_K; // 256
        int elems_per_thread = 8;
        int idx = lane_id * elems_per_thread;

        if (idx < elements_per_tile) {
            int row = idx / TILE_K;
            int col = idx % TILE_K;

            int a_row = c_row + row;
            int a_col = 0 + col;

            const half* gA = A + a_row * K + a_col;
            half* sA = Asmem + buf * (TILE_M * TILE_K) + row * TILE_K + col;

            cp_async_16(sA, gA);
        }

        int elements_per_tile_B = TILE_K * TILE_N;
        if (idx < elements_per_tile_B) {
            int row = idx / TILE_N;
            int col = idx % TILE_N;

            int b_row = 0 + row;
            int b_col = c_col + col;

            const half* gB = B + b_row * N + b_col;
            half* sB = Bsmem + buf * (TILE_K * TILE_N) + row * TILE_N + col;

            cp_async_16(sB, gB);
        }

        cp_async_commit();
    }

    cp_async_wait();
    __syncthreads();

#pragma unroll
    for (int tk = 0; tk < num_k_tiles; ++tk) {
        half* sA = Asmem + buf * (TILE_M * TILE_K);
        half* sB = Bsmem + buf * (TILE_K * TILE_N);

        if (warp_id == 1) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

            wmma::load_matrix_sync(a_frag, sA, TILE_K);
            wmma::load_matrix_sync(b_frag, sB, TILE_N);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        int next_k = tk + 1;
        int next_buf = buf ^ 1;

        if (next_k < num_k_tiles && warp_id == 0) {
            int elements_per_tile = TILE_M * TILE_K;
            int elems_per_thread = 8;
            int idx = lane_id * elems_per_thread;

            if (idx < elements_per_tile) {
                int row = idx / TILE_K;
                int col = idx % TILE_K;

                int a_row = c_row + row;
                int a_col = next_k * TILE_K + col;

                const half* gA = A + a_row * K + a_col;
                half* sA2 = Asmem + next_buf * (TILE_M * TILE_K) + row * TILE_K + col;

                cp_async_16(sA2, gA);
            }

            int elements_per_tile_B = TILE_K * TILE_N;
            if (idx < elements_per_tile_B) {
                int row = idx / TILE_N;
                int col = idx % TILE_N;

                int b_row = next_k * TILE_K + row;
                int b_col = c_col + col;

                const half* gB = B + b_row * N + b_col;
                half* sB2 = Bsmem + next_buf * (TILE_K * TILE_N) + row * TILE_N + col;

                cp_async_16(sB2, gB);
            }

            cp_async_commit();
        }

        if (next_k < num_k_tiles) {
            cp_async_wait();
            __syncthreads();
            buf ^= 1;
        }
    }

    // compute warp가 C 저장
    if (warp_id == 1) {
        float* c_tile_ptr = C + c_row * N + c_col;
        wmma::store_matrix_sync(c_tile_ptr, c_frag, N, wmma::mem_row_major);
    }
}

// ------------------------
// Host helpers
// ------------------------
void init_matrix_half(half* hA, int rows, int cols, float scale = 1.0f) {
    for (int i = 0; i < rows * cols; ++i) {
        float v = (float)(rand() % 5) * scale;
        hA[i] = __float2half(v);
    }
}

float compute_checksum(const float* hC, int rows, int cols) {
    double sum = 0.0;
    for (int i = 0; i < rows * cols; ++i) {
        sum += hC[i];
    }
    return static_cast<float>(sum);
}

float benchmark_kernel(dim3 grid, dim3 block,
                       size_t shmem_bytes,
                       const half* dA,
                       const half* dB,
                       float* dC,
                       int iters,
                       bool use_ws)
{
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        if (use_ws) {
            cpasync_ws_kernel<<<grid, block, shmem_bytes>>>(dA, dB, dC);
        } else {
            cpasync_mono_kernel<<<grid, block, shmem_bytes>>>(dA, dB, dC);
        }
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / iters;
}

int main(int argc, char** argv)
{
    int iters = 100;
    if (argc > 1) {
        iters = std::atoi(argv[1]);
        if (iters <= 0) iters = 100;
    }

    printf("GEMM size: M=N=K=%d, WMMA tile=16x16, iters=%d\n", M, iters);

    size_t bytesA_half   = M * K * sizeof(half);
    size_t bytesB_half   = K * N * sizeof(half);
    size_t bytesC_float  = M * N * sizeof(float);

    half*  hA = (half*) std::malloc(bytesA_half);
    half*  hB = (half*) std::malloc(bytesB_half);
    float* hC = (float*)std::malloc(bytesC_float);

    std::srand(0);
    init_matrix_half(hA, M, K, 0.1f);
    init_matrix_half(hB, K, N, 0.2f);

    half*  dA;
    half*  dB;
    float* dC;
    CHECK_CUDA(cudaMalloc(&dA, bytesA_half));
    CHECK_CUDA(cudaMalloc(&dB, bytesB_half));
    CHECK_CUDA(cudaMalloc(&dC, bytesC_float));

    CHECK_CUDA(cudaMemcpy(dA, hA, bytesA_half, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytesB_half, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, bytesC_float));

    dim3 grid(BLOCK_COL_TILES, BLOCK_ROW_TILES);

    // shared mem: 2*(A_tile + B_tile) half
    size_t shmem_bytes =
        (2 * TILE_M * TILE_K + 2 * TILE_K * TILE_N) * sizeof(half);

    // monolithic: 1 warp
    dim3 block_mono(32, 1, 1);
    float t_mono = benchmark_kernel(grid, block_mono, shmem_bytes,
                                    dA, dB, dC, iters, false);
    CHECK_CUDA(cudaMemcpy(hC, dC, bytesC_float, cudaMemcpyDeviceToHost));
    float checksum_mono = compute_checksum(hC, M, N);

    // warp specialization: 2 warps (warp0: load, warp1: compute)
    CHECK_CUDA(cudaMemset(dC, 0, bytesC_float));
    dim3 block_ws(64, 1, 1);
    float t_ws = benchmark_kernel(grid, block_ws, shmem_bytes,
                                  dA, dB, dC, iters, true);
    CHECK_CUDA(cudaMemcpy(hC, dC, bytesC_float, cudaMemcpyDeviceToHost));
    float checksum_ws = compute_checksum(hC, M, N);

    printf("checksum mono = %.6f\n", checksum_mono);
    printf("checksum ws   = %.6f\n", checksum_ws);
    printf("\n=== Timing (avg over %d iters) ===\n", iters);
    printf("cp.async + MMA (mono)        : %.4f ms\n", t_mono);
    printf("cp.async + MMA (warp-spec)   : %.4f ms\n", t_ws);
    printf("Speedup (mono / ws)          : %.2fx\n", t_mono / t_ws);

    std::free(hA);
    std::free(hB);
    std::free(hC);
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    return 0;
}

/*
빌드:

nvcc -O3 -std=c++17 -arch=sm_86 warp_specialization_cpasync_test.cu -o warp_specialization_cpasync_test.exe

프로파일 예시:

ncu --kernel-name regex:cpasync_mono_kernel.*     --set full     --launch-skip 5 --launch-count 1     ./warp_specialization_cpasync_test.exe 100
ncu --kernel-name regex:cpasync_ws_kernel.*     --set full     --launch-skip 5 --launch-count 1     ./warp_specialization_cpasync_test.exe 100
또는 ws 커널도 regex에 추가해서 비교.
*/
