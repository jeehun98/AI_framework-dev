// wmma_fragment_load_swizzle_test.cu
// Compile example:
// nvcc -O3 -arch=sm_80 wmma_fragment_load_swizzle_test.cu -o wmma_fragment_load_swizzle_test.exe

#include <cstdio>
#include <cuda.h>
#include <mma.h>

using namespace nvcuda;

constexpr int WARP_SIZE      = 32;
constexpr int TILE_M         = 16;
constexpr int TILE_N         = 16;
constexpr int TILE_K         = 16;
constexpr int THREADS_PER_BLOCK = 32;   // 1 warp
constexpr int BLOCKS             = 80;  // 80 warps
constexpr int ITERS              = 4096; // 반복 횟수 (bank conflict 효과 키우기)

// =============================
// Kernel: normal fragment load
// =============================
__global__ void mma_fragment_normal_load_kernel(const half* __restrict__ A,
                                                const half* __restrict__ B,
                                                float* __restrict__ C,
                                                int lda, int ldb, int ldc)
{
    // 1 warp per block
    int lane_id = threadIdx.x; // 0..31

    // shared: padding 없는 타일
    extern __shared__ half shared_mem[];
    half* shA = shared_mem;
    half* shB = shared_mem + TILE_M * TILE_K; // 바로 뒤에 붙여서 사용

    // WMMA fragment
    wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // 전역에서는 1개의 16x16 타일만 반복 사용 (A, B는 16x16로 가정)
    const half* tileA = A;
    const half* tileB = B;

    for (int it = 0; it < ITERS; ++it) {
        // --- global -> shared: 16x16 타일 로드 ---
        // 16*16 = 256 half elements, 32 thread * 8 = 256
        for (int i = 0; i < 8; ++i) {
            int idx = lane_id + i * WARP_SIZE;  // 0..255
            int row = idx / TILE_K;
            int col = idx % TILE_K;

            shA[row * TILE_K + col] = tileA[row * lda + col];
            shB[row * TILE_N + col] = tileB[row * ldb + col];
        }

        __syncthreads();

        // --- shared -> fragment (bank conflict 가능성 높은 layout) ---
        // stride = TILE_K (=16)
        wmma::load_matrix_sync(a_frag, shA, TILE_K);
        wmma::load_matrix_sync(b_frag, shB, TILE_N);

        // Tensor Core 연산
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // 결과를 하나만 써서 dead-code 제거 방지
    if (lane_id == 0) {
        C[blockIdx.x] = c_frag.x[0];
    }
}

// ===================================
// Kernel: swizzled / padded fragment load
// ===================================
__global__ void mma_fragment_swizzled_load_kernel(const half* __restrict__ A,
                                                  const half* __restrict__ B,
                                                  float* __restrict__ C,
                                                  int lda, int ldb, int ldc)
{
    int lane_id = threadIdx.x; // 0..31

    // padding 추가: row stride를 16 -> 18 elements로 변경
    constexpr int PAD = 2;
    constexpr int STRIDE_A = TILE_K + PAD; // 18
    constexpr int STRIDE_B = TILE_N + PAD; // 18

    extern __shared__ half shared_mem[];
    half* shA = shared_mem;
    half* shB = shared_mem + TILE_M * STRIDE_A;

    wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    const half* tileA = A;
    const half* tileB = B;

    for (int it = 0; it < ITERS; ++it) {
        // --- global -> shared (row padding 포함) ---
        // 16x16 타일을 16x(16+PAD) 버퍼에 채움
        for (int i = 0; i < 8; ++i) {
            int idx = lane_id + i * WARP_SIZE;  // 0..255
            int row = idx / TILE_K;
            int col = idx % TILE_K;

            // padded row stride
            shA[row * STRIDE_A + col] = tileA[row * lda + col];
            shB[row * STRIDE_B + col] = tileB[row * ldb + col];
        }

        __syncthreads();

        // --- shared -> fragment ---
        // stride = STRIDE_A / STRIDE_B (18) : row 크기가 32B의 배수가 아니어서
        // bank mapping이 달라지고 conflict가 줄어든다.
        wmma::load_matrix_sync(a_frag, shA, STRIDE_A);
        wmma::load_matrix_sync(b_frag, shB, STRIDE_B);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    if (lane_id == 0) {
        C[blockIdx.x] = c_frag.x[0];
    }
}

// =============================
// Host utility
// =============================
void check_cuda(cudaError_t e, const char* msg)
{
    if (e != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
        std::exit(EXIT_FAILURE);
    }
}

int main()
{
    printf("=== MMA / WMMA Test 2: Fragment load (normal vs swizzled) ===\n");
    printf("Tile: 16x16x16, BLOCKS=%d, THREADS_PER_BLOCK=%d, ITERS=%d\n",
           BLOCKS, THREADS_PER_BLOCK, ITERS);

    // 16x16 타일 하나만 준비 (모든 warp가 같은 타일을 반복 사용)
    const int M = TILE_M;
    const int N = TILE_N;
    const int K = TILE_K;
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    size_t bytesA = sizeof(half) * M * K;
    size_t bytesB = sizeof(half) * K * N;
    size_t bytesC = sizeof(float) * BLOCKS;

    half* hA = (half*)malloc(bytesA);
    half* hB = (half*)malloc(bytesB);
    float* hC = (float*)malloc(bytesC);

    // 간단 초기화
    for (int i = 0; i < M * K; ++i) {
        float v = (i % 7) * 0.1f;
        hA[i] = __float2half(v);
    }
    for (int i = 0; i < K * N; ++i) {
        float v = (i % 5) * 0.2f;
        hB[i] = __float2half(v);
    }

    half* dA;
    half* dB;
    float* dC;

    check_cuda(cudaMalloc(&dA, bytesA), "cudaMalloc dA");
    check_cuda(cudaMalloc(&dB, bytesB), "cudaMalloc dB");
    check_cuda(cudaMalloc(&dC, bytesC), "cudaMalloc dC");

    check_cuda(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice), "copy A");
    check_cuda(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice), "copy B");

    // shared memory 크기 계산
    size_t shmem_normal = sizeof(half) * (TILE_M * TILE_K * 2);              // A,B 16x16
    size_t shmem_swiz   = sizeof(half) * (TILE_M * (TILE_K + 2) * 2);        // A,B 16x(16+2)

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // -----------------------------
    // 1) normal fragment load
    // -----------------------------
    check_cuda(cudaMemset(dC, 0, bytesC), "memset C (normal)");

    cudaEventRecord(start);
    mma_fragment_normal_load_kernel<<<BLOCKS, THREADS_PER_BLOCK, shmem_normal>>>(
        dA, dB, dC, lda, ldb, ldc);
    cudaEventRecord(stop);

    check_cuda(cudaGetLastError(), "kernel launch (normal)");
    cudaEventSynchronize(stop);

    float time_ms_normal = 0.0f;
    cudaEventElapsedTime(&time_ms_normal, start, stop);

    check_cuda(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost), "copy C (normal)");

    printf("[normal]   kernel time: %.3f ms, sample C[0]=%f\n",
           time_ms_normal, hC[0]);

    // -----------------------------
    // 2) swizzled fragment load
    // -----------------------------
    check_cuda(cudaMemset(dC, 0, bytesC), "memset C (swizzled)");

    cudaEventRecord(start);
    mma_fragment_swizzled_load_kernel<<<BLOCKS, THREADS_PER_BLOCK, shmem_swiz>>>(
        dA, dB, dC, lda, ldb, ldc);
    cudaEventRecord(stop);

    check_cuda(cudaGetLastError(), "kernel launch (swizzled)");
    cudaEventSynchronize(stop);

    float time_ms_swiz = 0.0f;
    cudaEventElapsedTime(&time_ms_swiz, start, stop);

    check_cuda(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost), "copy C (swizzled)");

    printf("[swizzled] kernel time: %.3f ms, sample C[0]=%f\n",
           time_ms_swiz, hC[0]);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA);
    free(hB);
    free(hC);

    return 0;
}
/*
# normal fragment load
ncu --kernel-name regex:mma_fragment_normal_load_kernel.* --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed     ./wmma_fragment_load_swizzle_test.exe

# swizzled fragment load
ncu --kernel-name regex:mma_fragment_swizzled_load_kernel.* --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed     ./wmma_fragment_load_swizzle_test.exe


*/