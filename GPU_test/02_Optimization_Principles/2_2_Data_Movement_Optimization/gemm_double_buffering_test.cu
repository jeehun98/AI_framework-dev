#include <cstdio>
#include <cstdlib>
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
// 설정
// =======================
// C[M x N] = A[M x K] * B[K x N]
constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;

// 타일 크기 (block tile = 16x16, K tile = 16)
constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 16;

// =======================
// 호스트 유틸
// =======================
void init_matrix(float* a, int rows, int cols, float scale = 1.0f) {
    for (int i = 0; i < rows * cols; ++i) {
        a[i] = scale * ((i % 13) - 6); // -6 ~ 6 사이 값
    }
}

void gemm_host_ref(const float* A, const float* B, float* C,
                   int M, int N, int K)
{
    // A: [M x K], row-major
    // B: [K x N], row-major
    // C: [M x N], row-major
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

float max_abs_diff(const float* ref, const float* test, int n) {
    float maxd = 0.0f;
    for (int i = 0; i < n; ++i) {
        float d = fabsf(ref[i] - test[i]);
        if (d > maxd) maxd = d;
    }
    return maxd;
}

// =======================
// Single Buffer GEMM
// =======================
// shared A[16x16], B[16x16], 매 iteration마다:
//   load → __syncthreads → compute → __syncthreads
__global__ void gemm_single_buffer_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    int tx = threadIdx.x; // 0..15
    int ty = threadIdx.y; // 0..15

    int row = blockIdx.y * TILE_M + ty;
    int col = blockIdx.x * TILE_N + tx;

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // load A tile [block-row, k0..k0+15]
        if (row < M && (k0 + tx) < K)
            As[ty][tx] = A[row * K + (k0 + tx)];
        else
            As[ty][tx] = 0.0f;

        // load B tile [k0..k0+15, block-col]
        if ((k0 + ty) < K && col < N)
            Bs[ty][tx] = B[(k0 + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads(); // load 완료 대기

        // compute tile
        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            acc += As[ty][kk] * Bs[kk][tx];
        }

        __syncthreads(); // 다음 iteration에서 shared를 덮어쓰기 전에 동기화
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// =======================
// Double Buffer GEMM
// =======================
// shared A[2][16x16], B[2][16x16] 두 개 버퍼를 번갈아 사용.
// 패턴:
//   prologue: tile0 load → __syncthreads()
//   loop:
//     compute using read_buf
//     prefetch next tile into write_buf
//     __syncthreads()  // 다음 iteration에서 read_buf로 쓰기 전에 sync
//   → tile당 membar 1번으로 감소
__global__ void gemm_double_buffer_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[2][TILE_M][TILE_K];
    __shared__ float Bs[2][TILE_K][TILE_N];

    int tx = threadIdx.x; // 0..15
    int ty = threadIdx.y; // 0..15

    int row = blockIdx.y * TILE_M + ty;
    int col = blockIdx.x * TILE_N + tx;

    float acc = 0.0f;

    const int num_tiles = (K + TILE_K - 1) / TILE_K;

    int read_buf  = 0;
    int write_buf = 1;

    // ------------------
    // Prologue: tile 0 load
    // ------------------
    if (num_tiles > 0) {
        int k0 = 0;
        if (row < M && (k0 + tx) < K)
            As[read_buf][ty][tx] = A[row * K + (k0 + tx)];
        else
            As[read_buf][ty][tx] = 0.0f;

        if ((k0 + ty) < K && col < N)
            Bs[read_buf][ty][tx] = B[(k0 + ty) * N + col];
        else
            Bs[read_buf][ty][tx] = 0.0f;
    }

    __syncthreads(); // tile0 준비 완료

    // ------------------
    // Main loop
    // ------------------
    for (int tile = 0; tile < num_tiles; ++tile) {
        int k0 = tile * TILE_K;

        // 1) 현재 tile(read_buf)로 compute
        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            // 마지막 tile에서 K가 TILE_K의 배수가 아니라면, 불필요한 kk도 돌지만
            // As/Bs에 이미 0 채워져 있어서 안전.
            acc += As[read_buf][ty][kk] * Bs[read_buf][kk][tx];
        }

        // 2) 다음 tile을 write_buf로 prefetch
        int next_tile = tile + 1;
        if (next_tile < num_tiles) {
            int k_next = next_tile * TILE_K;

            if (row < M && (k_next + tx) < K)
                As[write_buf][ty][tx] = A[row * K + (k_next + tx)];
            else
                As[write_buf][ty][tx] = 0.0f;

            if ((k_next + ty) < K && col < N)
                Bs[write_buf][ty][tx] = B[(k_next + ty) * N + col];
            else
                Bs[write_buf][ty][tx] = 0.0f;
        }

        // 3) 다음 iteration에서 write_buf를 read_buf로 쓰기 위해 sync
        __syncthreads();

        // 4) buffer swap
        int tmp    = read_buf;
        read_buf   = write_buf;
        write_buf  = tmp;
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// =======================
// main
// =======================
int main()
{
    printf("=== Double Buffering Test 1: Single vs Double Buffer GEMM ===\n");
    printf("GEMM config: C[%d x %d] = A[%d x %d] * B[%d x %d]\n",
           M, N, M, K, K, N);
    printf("Tile: TILE_M=%d, TILE_N=%d, TILE_K=%d\n\n",
           TILE_M, TILE_N, TILE_K);

    size_t bytesA = sizeof(float) * M * K;
    size_t bytesB = sizeof(float) * K * N;
    size_t bytesC = sizeof(float) * M * N;

    float *hA = (float*)malloc(bytesA);
    float *hB = (float*)malloc(bytesB);
    float *hC_ref  = (float*)malloc(bytesC);
    float *hC_single = (float*)malloc(bytesC);
    float *hC_double = (float*)malloc(bytesC);

    init_matrix(hA, M, K, 0.01f);
    init_matrix(hB, K, N, 0.02f);

    printf("Computing host reference...\n");
    gemm_host_ref(hA, hB, hC_ref, M, N, K);

    float *dA, *dB, *dC_single, *dC_double;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC_single, bytesC));
    CHECK_CUDA(cudaMalloc(&dC_double, bytesC));

    CHECK_CUDA(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

    dim3 block(TILE_N, TILE_M); // (16,16)
    dim3 grid((N + TILE_N - 1) / TILE_N,
              (M + TILE_M - 1) / TILE_M);

    // -----------------------
    // 1) Single buffer
    // -----------------------
    {
        printf("\n[single_buffer] warm-up + timing\n");
        gemm_single_buffer_kernel<<<grid, block>>>(dA, dB, dC_single, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        gemm_single_buffer_kernel<<<grid, block>>>(dA, dB, dC_single, M, N, K);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        CHECK_CUDA(cudaMemcpy(hC_single, dC_single, bytesC, cudaMemcpyDeviceToHost));
        float diff = max_abs_diff(hC_ref, hC_single, M * N);

        printf("[single_buffer] time: %.3f ms, max diff = %e\n", ms, diff);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    // -----------------------
    // 2) Double buffer
    // -----------------------
    {
        printf("\n[double_buffer] warm-up + timing\n");
        gemm_double_buffer_kernel<<<grid, block>>>(dA, dB, dC_double, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        gemm_double_buffer_kernel<<<grid, block>>>(dA, dB, dC_double, M, N, K);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        CHECK_CUDA(cudaMemcpy(hC_double, dC_double, bytesC, cudaMemcpyDeviceToHost));
        float diff = max_abs_diff(hC_ref, hC_double, M * N);

        printf("[double_buffer] time: %.3f ms, max diff = %e\n", ms, diff);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC_single));
    CHECK_CUDA(cudaFree(dC_double));

    free(hA);
    free(hB);
    free(hC_ref);
    free(hC_single);
    free(hC_double);

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}
/*
nvcc -O3 -arch=sm_86 -lineinfo -o gemm_double_buffering_test.exe gemm_double_buffering_test.cu

ncu --kernel-name regex:gemm_single_buffer_kernel.* --metrics smsp__warp_issue_stalled_membar_per_warp_active.avg,smsp__warp_issue_stalled_membar_per_warp_active.pct,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ./gemm_double_buffering_test.exe

ncu --kernel-name regex:gemm_double_buffer_kernel.* --metrics smsp__warp_issue_stalled_membar_per_warp_active.avg,smsp__warp_issue_stalled_membar_per_warp_active.pct,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ./gemm_double_buffering_test.exe

*/