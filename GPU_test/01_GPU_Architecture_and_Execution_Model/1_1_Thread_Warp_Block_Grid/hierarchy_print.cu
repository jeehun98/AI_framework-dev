#include <cstdio>
#include <cuda_runtime.h>

constexpr int THREADS_PER_BLOCK = 128;  // 4 warps per block
constexpr int BLOCKS            = 3;    // 3 blocks in grid

// ------------------------------------------------------------
// Kernel: Block / Warp / Thread 계층 구조 기록
// ------------------------------------------------------------
__global__
void hierarchy_print(int* warp_out, int* block_out)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int warp_id = tid / 32;  // block 내 warp index (0~3)

    // 전역 인덱스: block 단위로 뒤에 이어붙이기
    int global_thread_index = bid * blockDim.x + tid;

    warp_out[global_thread_index] = warp_id;      // 이 thread가 어느 warp인지
    block_out[bid]                = blockDim.x / 32; // block 내 warp 수(=4)
}

// ------------------------------------------------------------
// Host 코드
// ------------------------------------------------------------
int main()
{
    const int num_threads_total = THREADS_PER_BLOCK * BLOCKS; // 128 * 3 = 384

    int* d_warp_out;
    int* d_block_out;

    cudaMalloc(&d_warp_out,  num_threads_total * sizeof(int));
    cudaMalloc(&d_block_out, BLOCKS * sizeof(int));

    hierarchy_print<<<BLOCKS, THREADS_PER_BLOCK>>>(d_warp_out, d_block_out);
    cudaDeviceSynchronize();

    int h_warp_out[num_threads_total];
    int h_block_out[BLOCKS];

    cudaMemcpy(h_warp_out,  d_warp_out,  num_threads_total * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_block_out, d_block_out, BLOCKS * sizeof(int),            cudaMemcpyDeviceToHost);

    printf("== Block / Warp / Thread mapping ==\n\n");

    for (int b = 0; b < BLOCKS; ++b) {
        printf("Block %d: warps_per_block = %d\n", b, h_block_out[b]);

        for (int t = 0; t < THREADS_PER_BLOCK; ++t) {
            int gtid    = b * THREADS_PER_BLOCK + t;
            int warp_id = h_warp_out[gtid];

            if (t % 32 == 0) {
                printf("  Warp %d: ", warp_id);
            }

            printf("%2d ", t);

            if (t % 32 == 31) {
                printf("\n");
            }
        }
        printf("\n");
    }

    cudaFree(d_warp_out);
    cudaFree(d_block_out);
    return 0;
}

// nvcc -O3 -arch=sm_86 hierarchy_print.cu -o hierarchy_print.exe

// ncu --set full --kernel-name "hierarchy_print" ./hierarchy_print.exe

