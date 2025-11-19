// kernel_hierarchy_test.cu
#include <cstdio>
#include <cuda_runtime.h>

struct ThreadInfo {
    int block_x, block_y;
    int thread_x, thread_y;
    int warp_in_block;
    int lane_in_warp;
    int global_row;
    int global_col;
};

__global__ void hierarchy_test_kernel(ThreadInfo* info,
                                      int rows, int cols)
{
    // ----- [1] 2D 전역 인덱스 (행렬 좌표라고 생각하면 됨) -----
    int global_row = blockIdx.y * blockDim.y + threadIdx.y;
    int global_col = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_row >= rows || global_col >= cols) return;

    // ----- [2] 블록 내 선형 thread 인덱스 -----
    int tid_in_block = threadIdx.y * blockDim.x + threadIdx.x;

    // ----- [3] warp / lane 계산 -----
    int warp_size = warpSize; // 보통 32
    int warp_in_block = tid_in_block / warp_size;
    int lane_in_warp = tid_in_block % warp_size;

    // ----- [4] 전역 선형 인덱스 (info 배열 인덱스) -----
    int idx = global_row * cols + global_col;

    // ----- [5] 이 커널 코드 전체는 "단일 스레드" 기준 동작 정의 -----
    ThreadInfo ti;
    ti.block_x       = blockIdx.x;
    ti.block_y       = blockIdx.y;
    ti.thread_x      = threadIdx.x;
    ti.thread_y      = threadIdx.y;
    ti.warp_in_block = warp_in_block;
    ti.lane_in_warp  = lane_in_warp;
    ti.global_row    = global_row;
    ti.global_col    = global_col;

    info[idx] = ti;
}

int main()
{
    // 테스트용 가벼운 2D 크기
    const int rows = 8;
    const int cols = 8;

    // 블록 크기 (8x8 = 64 threads, 블록당 2 warps)
    dim3 blockDim(8, 8);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x,
                 (rows + blockDim.y - 1) / blockDim.y);

    int num_elems = rows * cols;

    ThreadInfo* d_info = nullptr;
    ThreadInfo* h_info = new ThreadInfo[num_elems];

    cudaMalloc(&d_info, num_elems * sizeof(ThreadInfo));

    hierarchy_test_kernel<<<gridDim, blockDim>>>(d_info, rows, cols);
    cudaDeviceSynchronize();

    cudaMemcpy(h_info, d_info,
               num_elems * sizeof(ThreadInfo),
               cudaMemcpyDeviceToHost);

    // 결과 몇 개만 출력
    printf("=== Thread hierarchy dump (rows=%d, cols=%d) ===\n", rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int idx = r * cols + c;
            const ThreadInfo& ti = h_info[idx];
            printf("[global (%2d,%2d)]  block(%d,%d)  thread(%d,%d)  warp=%d  lane=%2d\n",
                   ti.global_row, ti.global_col,
                   ti.block_x, ti.block_y,
                   ti.thread_x, ti.thread_y,
                   ti.warp_in_block, ti.lane_in_warp);
        }
        printf("----------------------------------------------------------\n");
    }

    cudaFree(d_info);
    delete[] h_info;

    return 0;
}

// nvcc -O2 kernel_hierarchy_test.cu -o kernel_hierarchy_test