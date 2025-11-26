#include <cstdio>
#include <cuda_runtime.h>

constexpr int THREADS_PER_BLOCK = 128;   // 4 warps
constexpr int BLOCKS             = 1;    // 단일 블록만 사용

// ------------------------------------------------------------
// 1) Warp divergence 발생 커널
//    같은 warp 안에서 lane_id < 16 / >= 16 이 갈라지는 패턴
// ------------------------------------------------------------
__global__
void warp_divergence_kernel(int* out, int iters)
{
    int tid     = threadIdx.x;
    int lane_id = tid % 32;

    int acc = 0;

    for (int i = 0; i < iters; ++i) {
        if (lane_id < 16) {
            // 경로 A
            acc += lane_id;
        } else {
            // 경로 B
            acc += lane_id * 2;
        }
    }

    out[tid] = acc;
}

// ------------------------------------------------------------
// 2) Warp-uniform branch 버전
//    warp_id 기준으로 분기 → warp 내에서는 모두 같은 경로
// ------------------------------------------------------------
__global__
void warp_uniform_kernel(int* out, int iters)
{
    int tid     = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int acc = 0;

    for (int i = 0; i < iters; ++i) {
        if (warp_id < 2) {
            // warp 0,1 전체가 경로 A
            acc += lane_id;
        } else {
            // warp 2,3 전체가 경로 B
            acc += lane_id * 2;
        }
    }

    out[tid] = acc;
}

// ------------------------------------------------------------
// Host 코드
// ------------------------------------------------------------
int main()
{
    const int num_threads = THREADS_PER_BLOCK;
    const int iters       = 1'000'000;

    int* d_out;
    cudaMalloc(&d_out, num_threads * sizeof(int));

    printf("== warp_divergence_kernel ==\n");
    warp_divergence_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_out, iters);
    cudaDeviceSynchronize();

    int h_out_div[num_threads];
    cudaMemcpy(h_out_div, d_out, num_threads * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sample outputs (divergent):\n");
    for (int i = 0; i < 32; ++i) {
        printf("tid %2d -> %d\n", i, h_out_div[i]);
    }

    printf("\n== warp_uniform_kernel ==\n");
    warp_uniform_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_out, iters);
    cudaDeviceSynchronize();

    int h_out_uni[num_threads];
    cudaMemcpy(h_out_uni, d_out, num_threads * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sample outputs (uniform):\n");
    for (int i = 0; i < 32; ++i) {
        printf("tid %2d -> %d\n", i, h_out_uni[i]);
    }

    cudaFree(d_out);
    return 0;
}

// nvcc -O3 -arch=sm_86 warp_divergence_test.cu -o warp_divergence_test.exe
// ncu --set full --kernel-name "warp_divergence_kernel" ./warp_divergence_test.exe