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
// 설정값
// =======================

constexpr int THREADS_PER_BLOCK = 256;   // 8 warps per block
constexpr int BLOCKS            = 80;    // 총 warp = 640
constexpr int WARP_SIZE         = 32;
constexpr int WARPS_PER_BLOCK   = THREADS_PER_BLOCK / WARP_SIZE;
constexpr int TOTAL_WARPS       = BLOCKS * WARPS_PER_BLOCK;  // 640

constexpr int ITERS             = 1024;

// “행”/“열” 해석용 매트릭스 크기
//   ROWS = warp 수
//   COLS = ITERS * 32
constexpr int ROWS              = TOTAL_WARPS;                 // 640
constexpr int COLS              = ITERS * WARP_SIZE;           // 1024 * 32 = 32768

// col-major + padding 용 leading dimension
// (논리 ROWS=640인데, LD=641로 padding)
constexpr int LD_COL_PAD        = ROWS + 1;                    // 641

// =======================
// 유틸
// =======================

void init_row_major(float* buf, int rows, int cols, float scale = 1.0f) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int idx = r * cols + c;
            buf[idx] = scale * ((idx % 1024) * 0.001f);
        }
    }
}

void init_col_major_padded_from_row_major(const float* row_major,
                                          float* col_major_pad,
                                          int rows, int cols, int ld_col)
{
    // row_major: [rows x cols], row-major
    // col_major_pad: [ld_col x cols], col-major (ld_col >= rows)
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float v = row_major[r * cols + c];
            col_major_pad[c * ld_col + r] = v;
        }
    }
}

// =======================
// 1) row-major + row 방향 access (coalesced)
//    - threadIdx.x (lane)가 같은 row에서 col 방향으로 움직임
// =======================
__global__ void row_major_row_access_kernel(const float* __restrict__ in,
                                            float* __restrict__ out,
                                            int iters, int rows, int cols)
{
    int tid      = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id  = tid / 32;         // 0..(TOTAL_WARPS-1)
    int lane     = threadIdx.x & 31; // 0..31

    if (warp_id >= rows) return;

    int row = warp_id;

    float acc = 0.0f;

    // iters 동안 col 방향으로 전진, lane 끼리 연속된 col에 접근
    for (int it = 0; it < iters; ++it) {
        int col = it * 32 + lane;  // lane=0..31 → col 연속
        if (col < cols) {
            int idx = row * cols + col;  // row-major, 연속 col
            float v = in[idx];
            acc += v;
        }
    }

    out[tid] = acc;
}

// =======================
// 2) row-major + col 방향 access (mismatch)
//    - threadIdx.x (lane)가 같은 col에서 row 방향으로 움직이는 꼴
//    - stride ≈ cols (매우 큰 stride)
// =======================
__global__ void row_major_col_mismatch_kernel(const float* __restrict__ in,
                                              float* __restrict__ out,
                                              int iters, int rows, int cols)
{
    int tid      = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id  = tid / 32;
    int lane     = threadIdx.x & 31;

    if (warp_id >= cols) return; // col index로 warp_id 사용

    int col = warp_id;   // col 고정
    float acc = 0.0f;

    // lane이 row 방향으로 움직임 → stride = cols
    for (int it = 0; it < iters; ++it) {
        int row = it * 32 + lane;  // lane=0..31 → row가 멀리 떨어져 있음
        if (row < rows) {
            int idx = row * cols + col;  // row-major, row가 앞에 있음 → stride=cols
            float v = in[idx];
            acc += v;
        }
    }

    out[tid] = acc;
}

// =======================
// 3) col-major + padding + col 방향 access (복구)
//    - 메모리에 col-major(+padding)로 저장한 버퍼
//    - threadIdx.x가 row 방향으로 움직이지만, 실제로는 연속된 메모리
//      idx = col * ld + row → row 증가에 따라 주소 +1
// =======================
__global__ void col_major_padded_col_access_kernel(const float* __restrict__ in_cm,
                                                   float* __restrict__ out,
                                                   int iters, int rows, int cols, int ld_col)
{
    int tid      = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id  = tid / 32;
    int lane     = threadIdx.x & 31;

    if (warp_id >= cols) return;

    int col = warp_id;
    float acc = 0.0f;

    for (int it = 0; it < iters; ++it) {
        int row = it * 32 + lane;
        if (row < rows) {
            int idx = col * ld_col + row;  // col-major, row 증가 → idx 연속
            float v = in_cm[idx];
            acc += v;
        }
    }

    out[tid] = acc;
}

// =======================
// main
// =======================
int main()
{
    printf("=== Coalescing Test 2: Row-major vs Col-major mismatch ===\n");
    printf("THREADS_PER_BLOCK = %d, BLOCKS = %d, TOTAL_WARPS = %d\n",
           THREADS_PER_BLOCK, BLOCKS, TOTAL_WARPS);
    printf("ROWS = %d, COLS = %d, ITERS = %d, LD_COL_PAD = %d\n\n",
           ROWS, COLS, ITERS, LD_COL_PAD);

    size_t elems_row_major      = static_cast<size_t>(ROWS) * COLS;
    size_t elems_col_major_pad  = static_cast<size_t>(LD_COL_PAD) * COLS;
    size_t total_threads        = static_cast<size_t>(THREADS_PER_BLOCK) * BLOCKS;

    size_t bytes_row_major      = elems_row_major * sizeof(float);
    size_t bytes_col_major_pad  = elems_col_major_pad * sizeof(float);
    size_t bytes_out            = total_threads * sizeof(float);

    float* h_row_major     = (float*)malloc(bytes_row_major);
    float* h_col_major_pad = (float*)malloc(bytes_col_major_pad);
    float* h_out           = (float*)malloc(bytes_out);

    init_row_major(h_row_major, ROWS, COLS);
    // col-major(+padding) 버퍼로 변환
    init_col_major_padded_from_row_major(h_row_major, h_col_major_pad,
                                         ROWS, COLS, LD_COL_PAD);

    float *d_row_major, *d_col_major_pad, *d_out;
    CHECK_CUDA(cudaMalloc(&d_row_major,     bytes_row_major));
    CHECK_CUDA(cudaMalloc(&d_col_major_pad, bytes_col_major_pad));
    CHECK_CUDA(cudaMalloc(&d_out,           bytes_out));

    CHECK_CUDA(cudaMemcpy(d_row_major,     h_row_major,     bytes_row_major,     cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_major_pad, h_col_major_pad, bytes_col_major_pad, cudaMemcpyHostToDevice));

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCKS);

    // ------------------------------------------------
    // 1) row-major + row 방향 access (baseline coalesced)
    // ------------------------------------------------
    {
        printf("[row_major_row] warm-up + timing\n");
        row_major_row_access_kernel<<<grid, block>>>(
            d_row_major, d_out, ITERS, ROWS, COLS);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        row_major_row_access_kernel<<<grid, block>>>(
            d_row_major, d_out, ITERS, ROWS, COLS);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        printf("[row_major_row] kernel time: %.3f ms\n\n", ms);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    // ------------------------------------------------
    // 2) row-major + col 방향 access (layout mismatch → non-coalesced)
    // ------------------------------------------------
    {
        printf("[row_major_col_mismatch] warm-up + timing\n");
        row_major_col_mismatch_kernel<<<grid, block>>>(
            d_row_major, d_out, ITERS, ROWS, COLS);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        row_major_col_mismatch_kernel<<<grid, block>>>(
            d_row_major, d_out, ITERS, ROWS, COLS);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        printf("[row_major_col_mismatch] kernel time: %.3f ms\n\n", ms);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    // ------------------------------------------------
    // 3) col-major(+padding) + col 방향 access (coalescing 회복)
    // ------------------------------------------------
    {
        printf("[col_major_padded] warm-up + timing\n");
        col_major_padded_col_access_kernel<<<grid, block>>>(
            d_col_major_pad, d_out, ITERS, ROWS, COLS, LD_COL_PAD);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        col_major_padded_col_access_kernel<<<grid, block>>>(
            d_col_major_pad, d_out, ITERS, ROWS, COLS, LD_COL_PAD);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        printf("[col_major_padded] kernel time: %.3f ms\n\n", ms);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    CHECK_CUDA(cudaFree(d_row_major));
    CHECK_CUDA(cudaFree(d_col_major_pad));
    CHECK_CUDA(cudaFree(d_out));

    free(h_row_major);
    free(h_col_major_pad);
    free(h_out);

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}
/*
nvcc -O3 -arch=sm_86 -lineinfo -o coalescing_layout_mismatch_test.exe coalescing_layout_mismatch_test.cu

ncu --kernel-name regex:row_major_row_access_kernel.* --metrics dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed   ./coalescing_layout_mismatch_test.exe

ncu --kernel-name regex:row_major_col_mismatch_kernel.* --metrics dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed   ./coalescing_layout_mismatch_test.exe

ncu --kernel-name regex:col_major_padded_col_access_kernel.* --metrics dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed   ./coalescing_layout_mismatch_test.exe


*/