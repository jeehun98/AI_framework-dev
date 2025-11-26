#include <cstdio>
#include <cuda_runtime.h>

constexpr int THREADS = 128;

__global__
void mem_stall_kernel(const float* __restrict__ in, float* __restrict__ out, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float acc = 0.f;

    // 랜덤 탐색 → L1/L2 miss 유도 → memory stall
    for (int i = 0; i < 4096; i++) {
        int idx = (tid * 97 + i * 57) & (N - 1);  // stride-random
        acc += in[idx];
    }

    out[tid] = acc;
}

__global__
void dep_stall_kernel(const float* __restrict__ in, float* __restrict__ out, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float acc = in[tid];  // dependency 시작점

    // dependency chain → scoreboard stall 강제
    for (int i = 0; i < 4096; i++) {
        acc = acc * 1.0000001f + 0.0000001f;  // fma에 완전 의존
    }

    out[tid] = acc;
}

__global__
void mixed_stall_kernel(const float* __restrict__ in, float* __restrict__ out, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float acc = in[tid];

    for (int i = 0; i < 4096; i++) {
        // dependency
        acc = acc * 1.0000001f + 0.0000001f;

        // random memory (stall)
        int idx = (tid * 17 + i * 31) & (N - 1);
        acc += in[idx];
    }

    out[tid] = acc;
}

int main() {
    const int N = 1 << 20;

    float *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    printf("== Warp Stall Breakdown Test ==\n");

    mem_stall_kernel <<<80, THREADS>>> (d_in, d_out, N);
    dep_stall_kernel <<<80, THREADS>>> (d_in, d_out, N);
    mixed_stall_kernel<<<80, THREADS>>> (d_in, d_out, N);

    cudaDeviceSynchronize();

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

// nvcc -O3 -arch=sm_86 warp_stall_test.cu -o warp_stall_test.exe
