#include <cstdio>
#include <cuda_runtime.h>

constexpr int THREADS_PER_BLOCK = 128;  // 4 warps
constexpr int BLOCKS            = 1;

// side-effect를 강제로 만들어 줄 글로벌 변수
__device__ unsigned long long g_sink = 0;

__global__
void warp_schedule_test(unsigned long long* out)
{
    int tid     = threadIdx.x;
    int warp_id = tid / 32;
    int lane    = tid % 32;

    __syncthreads();  // 측정 시작 시점 맞추기 (선택)

    unsigned long long start = clock64();

    if (warp_id == 0) {
        // warp 0만 무거운 연산 수행
        unsigned long long acc = (unsigned long long)(tid);

        // 꽤 무거운 루프: 연산 + atomic side-effect
        for (int i = 0; i < 200000; ++i) {
            acc = acc * 1664525ull + 1013904223ull;  // LCG 한 번
            if ((i & 0x3FFF) == 0 && lane == 0) {
                // 너무 자주 atomic하면 stall만 너무 커지니, 가끔씩만
                atomicAdd(&g_sink, acc);
            }
        }
    }

    unsigned long long end = clock64();
    unsigned long long elapsed = end - start;

    out[tid] = elapsed;
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

// 빌드 예시 (그냥 O3 유지):
// nvcc -O3 -arch=sm_86 warp-schedule.cu -o warp-schedule.exe
// nvcc -arch=sm_86 -Xptxas -O0 warp-schedule.cu -o warp-schedule.exe
