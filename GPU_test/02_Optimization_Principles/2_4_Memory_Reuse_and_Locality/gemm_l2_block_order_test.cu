#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA Error %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err__));        \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

constexpr int BLOCK_M = 16;
constexpr int BLOCK_N = 16;

__global__ void gemm_block_order_row_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int tiles_m, int tiles_n)
{
    int tile_id = blockIdx.x;
    int total_tiles = tiles_m * tiles_n;
    if (tile_id >= total_tiles) return;

    // row-major tile ordering: (bm, bn) = (tile_id / tiles_n, tile_id % tiles_n)
    int bm = tile_id / tiles_n;
    int bn = tile_id % tiles_n;

    int row = bm * BLOCK_M + threadIdx.y;
    int col = bn * BLOCK_N + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        float a = A[row * K + k];
        float b = B[k * N + col];
        acc += a * b;
    }
    C[row * N + col] = acc;
}

__global__ void gemm_block_order_col_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int tiles_m, int tiles_n)
{
    int tile_id = blockIdx.x;
    int total_tiles = tiles_m * tiles_n;
    if (tile_id >= total_tiles) return;

    // column-major tile ordering: (bm, bn) = (tile_id % tiles_m, tile_id / tiles_m)
    int bm = tile_id % tiles_m;
    int bn = tile_id / tiles_m;

    int row = bm * BLOCK_M + threadIdx.y;
    int col = bn * BLOCK_N + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        float a = A[row * K + k];
        float b = B[k * N + col];
        acc += a * b;
    }
    C[row * N + col] = acc;
}

void host_gemm_ref(const float* A, const float* B, float* C,
                   int M, int N, int K)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

float max_abs_diff(const float* a, const float* b, int n)
{
    float maxd = 0.f;
    for (int i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > maxd) maxd = d;
    }
    return maxd;
}

int main()
{
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    const size_t sizeA = static_cast<size_t>(M) * K * sizeof(float);
    const size_t sizeB = static_cast<size_t>(K) * N * sizeof(float);
    const size_t sizeC = static_cast<size_t>(M) * N * sizeof(float);

    printf("=== L2 Locality Test 1: GEMM block ordering (row-major vs col-major tiles) ===\n");
    printf("GEMM: C[%d x %d] = A[%d x %d] * B[%d x %d]\n", M, N, M, K, K, N);
    printf("Block tile: BM=%d, BN=%d (1 element/thread, no SMEM)\n", BLOCK_M, BLOCK_N);

    // Host alloc
    float* h_A     = (float*)malloc(sizeA);
    float* h_B     = (float*)malloc(sizeB);
    float* h_C_ref = (float*)malloc(sizeC);
    float* h_C_row = (float*)malloc(sizeC);
    float* h_C_col = (float*)malloc(sizeC);

    if (!h_A || !h_B || !h_C_ref || !h_C_row || !h_C_col) {
        fprintf(stderr, "Host malloc failed\n");
        return EXIT_FAILURE;
    }

    // Init A,B
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = (float)((i % 17) - 8) * 0.125f;
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = (float)((i % 13) - 6) * 0.25f;
    }

    // Reference
    printf("Computing reference GEMM on host...\n");
    host_gemm_ref(h_A, h_B, h_C_ref, M, N, K);

    // Device alloc
    float *d_A, *d_B, *d_C_row, *d_C_col;
    CHECK_CUDA(cudaMalloc(&d_A, sizeA));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB));
    CHECK_CUDA(cudaMalloc(&d_C_row, sizeC));
    CHECK_CUDA(cudaMalloc(&d_C_col, sizeC));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    int tiles_m = (M + BLOCK_M - 1) / BLOCK_M;
    int tiles_n = (N + BLOCK_N - 1) / BLOCK_N;
    int total_tiles = tiles_m * tiles_n;

    dim3 block(BLOCK_N, BLOCK_M);      // (x=col, y=row) inside tile
    dim3 grid(total_tiles, 1, 1);

    printf("tiles_m=%d, tiles_n=%d, total_tiles=%d\n", tiles_m, tiles_n, total_tiles);
    printf("grid=(%d, %d, %d), block=(%d, %d, %d)\n",
           grid.x, grid.y, grid.z, block.x, block.y, block.z);

    // Events
    cudaEvent_t ev_start, ev_end;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_end));

    // -----------------------------
    // Row-major tile ordering run
    // -----------------------------
    printf("\n[Row-major block ordering] (bm,bn) = (tile_id / tiles_n, tile_id %% tiles_n)\n");
    CHECK_CUDA(cudaMemset(d_C_row, 0, sizeC));

    // Warm-up
    gemm_block_order_row_kernel<<<grid, block>>>(d_A, d_B, d_C_row, M, N, K, tiles_m, tiles_n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(ev_start));
    gemm_block_order_row_kernel<<<grid, block>>>(d_A, d_B, d_C_row, M, N, K, tiles_m, tiles_n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(ev_end));
    CHECK_CUDA(cudaEventSynchronize(ev_end));

    float ms_row = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_row, ev_start, ev_end));
    CHECK_CUDA(cudaMemcpy(h_C_row, d_C_row, sizeC, cudaMemcpyDeviceToHost));

    float max_diff_row = max_abs_diff(h_C_row, h_C_ref, M * N);
    printf("[Row-major] time = %.3f ms, max diff vs ref = %e\n",
           ms_row, max_diff_row);

    // -----------------------------
    // Col-major tile ordering run
    // -----------------------------
    printf("\n[Col-major block ordering] (bm,bn) = (tile_id %% tiles_m, tile_id / tiles_m)\n");
    CHECK_CUDA(cudaMemset(d_C_col, 0, sizeC));

    // Warm-up
    gemm_block_order_col_kernel<<<grid, block>>>(d_A, d_B, d_C_col, M, N, K, tiles_m, tiles_n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(ev_start));
    gemm_block_order_col_kernel<<<grid, block>>>(d_A, d_B, d_C_col, M, N, K, tiles_m, tiles_n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(ev_end));
    CHECK_CUDA(cudaEventSynchronize(ev_end));

    float ms_col = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_col, ev_start, ev_end));
    CHECK_CUDA(cudaMemcpy(h_C_col, d_C_col, sizeC, cudaMemcpyDeviceToHost));

    float max_diff_col = max_abs_diff(h_C_col, h_C_ref, M * N);
    printf("[Col-major] time = %.3f ms, max diff vs ref = %e\n",
           ms_col, max_diff_col);

    printf("\n=== Summary (timing only, profiling은 따로 ncu로) ===\n");
    printf("Row-major ordering time : %.3f ms\n", ms_row);
    printf("Col-major ordering time : %.3f ms\n", ms_col);
    if (ms_col > 0.f) {
        printf("Speed ratio (row / col) = %.2fx\n", ms_row / ms_col);
    }

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_end));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C_row));
    CHECK_CUDA(cudaFree(d_C_col));
    free(h_A);
    free(h_B);
    free(h_C_ref);
    free(h_C_row);
    free(h_C_col);

    return 0;
}
/*
nvcc -O3 gemm_l2_block_order_test.cu -o gemm_l2_block_order_test.exe

# Row-major tile ordering
ncu --kernel-name regex:gemm_block_order_row_kernel.*     --metrics lts__t_sectors_hit_rate.pct,dram__bytes_read.sum     ./gemm_l2_block_order_test.exe

# Col-major tile ordering
ncu --kernel-name regex:gemm_block_order_col_kernel.*     --metrics lts__t_sectors_hit_rate.pct,dram__bytes_read.sum     ./gemm_l2_block_order_test.exe


*/