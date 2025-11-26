#include <cstdio>
#include <cuda_runtime.h>

constexpr int THREADS_PER_BLOCK = 128;  // 4 warps
constexpr int BLOCKS            = 1;

__global__
void warp_schedule_test(unsigned long long* out)
{
    int tid     = threadIdx.x;
    int warp_id = tid / 32;
    int lane    = tid % 32;

    unsigned long long start = clock64();

    // warp 0에만 인위적 delay 부여
    if (warp_id == 0) {
        volatile int dummy = 0;  // 최적화 방지용
        for (int i = 0; i < 100000; ++i) {
            dummy += i;
        }
    }

    unsigned long long end = clock64();

    out[tid] = end - start;
}

int main()
{
    const int num_threads = THREADS_PER_BLOCK * BLOCKS;

    unsigned long long* d_out;
    cudaMalloc(&d_out, num_threads * sizeof(unsigned long long));

    warp_schedule_test<<<BLOCKS, THREADS_PER_BLOCK>>>(d_out);
    cudaDeviceSynchronize();

    unsigned long long h_out[num_threads];
    cudaMemcpy(h_out, d_out, num_threads * sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);

    printf("== warp schedule test ==\n\n");

    for (int w = 0; w < THREADS_PER_BLOCK / 32; ++w) {
        printf("Warp %d:\n  ", w);
        for (int lane = 0; lane < 32; ++lane) {
            int tid = w * 32 + lane;
            printf("%10llu ", h_out[tid]);
        }
        printf("\n\n");
    }

    cudaFree(d_out);
    return 0;
}

// 빌드 예시:
// nvcc -O3 -arch=sm_86 warp-schedule.cu -o warp-schedule.exe

// Nsight Compute 프로파일링 예시:
// ncu --set full --kernel-name "warp_schedule_test" ./warp-schedule.exe
