#include <cstdio>
#include <cuda_runtime.h>

constexpr int THREADS_PER_BLOCK = 128;
constexpr int BLOCKS            = 3;

// ------------------------------------------------------------
// Kernel: Block 단위 shared memory scope 확인
// ------------------------------------------------------------
__global__
void smem_scope_test(int* out)
{
    __shared__ int smem;

    if (threadIdx.x == 0)
        smem = blockIdx.x * 100;   // block별 고유 값

    __syncthreads();

    // block마다 서로 다른 smem 값을 찍고 싶은 것
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = smem;
}

// ------------------------------------------------------------
// Host 코드
// ------------------------------------------------------------
int main()
{
    const int num_threads_total = THREADS_PER_BLOCK * BLOCKS;

    int* d_out;
    cudaMalloc(&d_out, num_threads_total * sizeof(int));

    smem_scope_test<<<BLOCKS, THREADS_PER_BLOCK>>>(d_out);
    cudaDeviceSynchronize();

    int h_out[num_threads_total];
    cudaMemcpy(h_out, d_out, num_threads_total * sizeof(int), cudaMemcpyDeviceToHost);

    printf("== smem scope test ==\n\n");
    for (int b = 0; b < BLOCKS; ++b) {
        printf("Block %d:\n  ", b);
        for (int t = 0; t < THREADS_PER_BLOCK; ++t) {
            int idx = b * THREADS_PER_BLOCK + t;
            printf("%d ", h_out[idx]);
        }
        printf("\n\n");
    }

    cudaFree(d_out);
    return 0;
}

// nvcc -O3 -arch=sm_86 smem-scope-test.cu -o smem_scope_test.exe
// ncu --set full --kernel-name "smem_*" ./smem_bank_conflict_test.exe
