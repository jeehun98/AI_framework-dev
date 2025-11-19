// kernel_hierarchy_extended_test.cu
#include <cstdio>
#include <cuda_runtime.h>

// ---------------------------------------------
// [1] 계층 구조 관찰용 ThreadInfo
// ---------------------------------------------
struct ThreadInfo {
    int block_x, block_y;
    int thread_x, thread_y;
    int warp_in_block;
    int lane_in_warp;
    int global_row;
    int global_col;
};

// ---------------------------------------------
// [2] Shared Memory Bank 테스트용 BankInfo
// ---------------------------------------------
struct BankInfo {
    int tid;       // threadIdx.x
    int indexA;    // 패턴 A에서 접근한 shared index
    int bankA;     // 패턴 A에서의 bank 번호 (개념적)
    int indexB;    // 패턴 B에서 접근한 shared index
    int bankB;     // 패턴 B에서의 bank 번호 (개념적)
};

// ======================================================================
//  커널 1: Grid / Block / Warp / Thread 계층 구조 테스트
// ======================================================================
__global__ void hierarchy_test_kernel(ThreadInfo* info,
                                      int rows, int cols)
{
    // ----- [1] 2D 전역 인덱스 (행렬 좌표 처럼 사용) -----
    int global_row = blockIdx.y * blockDim.y + threadIdx.y;
    int global_col = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_row >= rows || global_col >= cols) return;

    // ----- [2] 블록 내 선형 thread 인덱스 -----
    int tid_in_block = threadIdx.y * blockDim.x + threadIdx.x;

    // ----- [3] warp / lane 계산 -----
    int warp_size = warpSize; // 일반적으로 32
    int warp_in_block = tid_in_block / warp_size;
    int lane_in_warp = tid_in_block % warp_size;

    // ----- [4] 전역 선형 인덱스 (info 배열 인덱스) -----
    int idx = global_row * cols + global_col;

    // ----- [5] 이 커널 전체는 "단일 스레드의 행동 정의" -----
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

// ======================================================================
//  커널 2: Shared Memory Bank 패턴 테스트
//
//  가정: 4-byte float, 32 banks → "index % 32"로 개념적 bank 번호 계산
// ======================================================================
__global__ void shared_mem_bank_test_kernel(BankInfo* out)
{
    // 여기서는 1D block, blockDim.x == 32 (1 warp) 가정
    extern __shared__ float smem[];

    int tid = threadIdx.x;

    // 패턴 A: indexA = tid
    // → 스레드 0~31 이 index 0~31 접근 → 개념적으로 bank 0~31 균등
    int indexA = tid;

    // 패턴 B: indexB = tid * 32
    // → 스레드 0~31 이 index 0, 32, 64, ... 접근
    //   indexB % 32 == 0 → 전부 bank 0 (개념적) → 심각한 bank conflict 패턴
    int indexB = tid * 32;

    // 실제로 shared memory에 접근 (의미는 크게 중요 X, 패턴 만드는 용도)
    smem[indexA] = (float)tid;
    smem[indexB] = (float)(tid * 2);

    // 개념적인 bank 번호 계산
    const int num_banks = 32; // 대부분 아키텍처에서 32개
    int bankA = (indexA % num_banks);
    int bankB = (indexB % num_banks);

    BankInfo bi;
    bi.tid    = tid;
    bi.indexA = indexA;
    bi.bankA  = bankA;
    bi.indexB = indexB;
    bi.bankB  = bankB;

    out[tid] = bi;
}

// ======================================================================
//  main: 두 커널을 순서대로 실행해서 결과 출력
// ======================================================================
int main()
{
    // -----------------------------
    // [파트 1] 계층 구조 테스트
    // -----------------------------
    {
        const int rows = 8;
        const int cols = 8;

        dim3 blockDim(8, 8); // 블록당 64 threads → warp 2개
        dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);

        int num_elems = rows * cols;

        ThreadInfo* d_info = nullptr;
        ThreadInfo* h_info = new ThreadInfo[num_elems];

        cudaMalloc(&d_info, num_elems * sizeof(ThreadInfo));

        hierarchy_test_kernel<<<gridDim, blockDim>>>(d_info, rows, cols);
        cudaDeviceSynchronize();

        cudaMemcpy(h_info, d_info,
                   num_elems * sizeof(ThreadInfo),
                   cudaMemcpyDeviceToHost);

        printf("=== [PART 1] Thread hierarchy dump (rows=%d, cols=%d) ===\n",
               rows, cols);
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
            printf("------------------------------------------------------------------\n");
        }

        cudaFree(d_info);
        delete[] h_info;
    }

    // -----------------------------
    // [파트 2] Shared Memory Bank 패턴 테스트
    // -----------------------------
    {
        const int block_threads = 32;  // 1 warp
        const int grid_blocks   = 1;

        dim3 blockDim(block_threads);
        dim3 gridDim(grid_blocks);

        BankInfo* d_bank = nullptr;
        BankInfo* h_bank = new BankInfo[block_threads];

        cudaMalloc(&d_bank, block_threads * sizeof(BankInfo));

        // shared memory 크기:
        // indexB 최대 값 = (block_threads-1) * 32
        // block_threads == 32 → max indexB = 31 * 32 = 992
        // 여유 있게 1024개 float 사용
        size_t shared_bytes = 1024 * sizeof(float);

        shared_mem_bank_test_kernel<<<gridDim, blockDim, shared_bytes>>>(d_bank);
        cudaDeviceSynchronize();

        cudaMemcpy(h_bank, d_bank,
                   block_threads * sizeof(BankInfo),
                   cudaMemcpyDeviceToHost);

        printf("\n=== [PART 2] Shared Memory Bank pattern test (blockDim.x=%d) ===\n",
               block_threads);
        printf("Assume: 32 banks, 4-byte float → bank = index %% 32 (개념적)\n\n");
        printf("tid | indexA bankA | indexB bankB\n");
        printf("----+--------------+--------------\n");
        for (int i = 0; i < block_threads; ++i) {
            const BankInfo& bi = h_bank[i];
            printf("%3d | %6d  %5d | %6d  %5d\n",
                   bi.tid,
                   bi.indexA, bi.bankA,
                   bi.indexB, bi.bankB);
        }

        cudaFree(d_bank);
        delete[] h_bank;
    }

    return 0;
}

//nvcc -O2 kernel_hierarchy_extended_test.cu -o kernel_hierarchy_extended_test