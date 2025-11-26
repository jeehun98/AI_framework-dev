#include <cstdio>
#include <cuda_runtime.h>

constexpr int THREADS = 128;   // 4 warps
constexpr int BLOCKS  = 1;

// ------------------------------------------------------------
// 실제 branch divergence가 발생하도록 강제한 커널
// (컴파일러 predication 최적화가 불가능하도록 만드는 형태)
// ------------------------------------------------------------
__global__
void forced_divergence_kernel(int* out)
{
    int tid     = threadIdx.x;
    int lane_id = tid % 32;

    long long acc = 0;

    // lane 0~15 vs lane 16~31 극단적으로 다른 연산량
    if (lane_id < 16) {
        // A-path: 가벼운 연산
        for (int i = 0; i < 1000; ++i) {
            acc += lane_id;
        }
    } else {
        // B-path: 매우 무거운 연산
        for (int i = 0; i < 10'000'000; ++i) {
            acc += lane_id * 2;
        }
    }

    out[tid] = acc;
}

// ------------------------------------------------------------
// warp-uniform branch 비교용 커널
// (warp 단위로 모두 같은 경로 → divergence 없음)
// ------------------------------------------------------------
__global__
void forced_uniform_kernel(int* out)
{
    int tid     = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    long long acc = 0;

    if (warp_id < 2) {
        for (int i = 0; i < 10'000'000; ++i) {
            acc += lane_id;
        }
    } else {
        for (int i = 0; i < 10'000'000; ++i) {
            acc += lane_id;
        }
    }

    out[tid] = acc;
}

// ------------------------------------------------------------
// Host
// ------------------------------------------------------------
int main()
{
    int* d_out;
    cudaMalloc(&d_out, THREADS * sizeof(int));

    printf("Running forced_divergence_kernel...\n");
    forced_divergence_kernel<<<BLOCKS, THREADS>>>(d_out);
    cudaDeviceSynchronize();

    int h_out_div[THREADS];
    cudaMemcpy(h_out_div, d_out, THREADS * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 32; ++i)
        printf("tid %2d -> %lld\n", i, (long long)h_out_div[i]);

    printf("\nRunning forced_uniform_kernel...\n");
    forced_uniform_kernel<<<BLOCKS, THREADS>>>(d_out);
    cudaDeviceSynchronize();

    int h_out_uni[THREADS];
    cudaMemcpy(h_out_uni, d_out, THREADS * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 32; ++i)
        printf("tid %2d -> %lld\n", i, (long long)h_out_uni[i]);

    cudaFree(d_out);
    return 0;
}

// nvcc -O3 -arch=sm_86 forced_warp_divergence_test.cu -o forced_warp_divergence_test.exe
// ncu --set full --kernel-name "forced_divergence_kernel" ./forced_warp_divergence_test.exe