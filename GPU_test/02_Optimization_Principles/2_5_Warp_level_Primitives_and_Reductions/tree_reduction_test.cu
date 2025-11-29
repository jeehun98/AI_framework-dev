#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr,"CUDA Error %s:%d : %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)
#endif

// =========================================
// Warp reduce (shfl_down)
// =========================================
__inline__ __device__
float warpReduce(float val) {
    unsigned mask = 0xffffffff;
    for(int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val; // lane 0에 최종 수렴
}

// =========================================
// Kernel 1: naive block reduction
// thread 0이 전체 thread 값을 loop로 더함
// =========================================
__global__
void reduce_naive(const float* __restrict__ g_in,
                  float* __restrict__ g_out,
                  int N)
{
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float val = (gid < N) ? g_in[gid] : 0.0f;
    smem[tid] = val;
    __syncthreads();

    float sum = 0.0f;

    // thread 0만 sequential loop 수행
    if (tid == 0) {
        for (int i = 0; i < blockDim.x; i++)
            sum += smem[i];
        g_out[blockIdx.x] = sum;
    }
}

// =========================================
// Kernel 2: warp-level tree reduction
// =========================================
__global__
void reduce_warp(const float* __restrict__ g_in,
                 float* __restrict__ g_out,
                 int N)
{
    extern __shared__ float warpSums[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float val = (gid < N) ? g_in[gid] : 0.0f;

    // 1) 각 warp 내부 reduction
    int lane   = tid & 31;
    int warpId = tid >> 5;

    val = warpReduce(val);

    // 2) warp당 결과를 shared에 기록
    if (lane == 0) warpSums[warpId] = val;
    __syncthreads();

    // 3) warp 0이 warpSums를 다시 reduction
    float warpVal = (tid < (blockDim.x / 32)) ? warpSums[lane] : 0.0f;

    if (warpId == 0) {
        float final = warpReduce(warpVal);
        if (lane == 0)
            g_out[blockIdx.x] = final;
    }
}

// =========================================
// CPU ref
// =========================================
float cpu_reduce(const float* h, int N)
{
    double s = 0;
    for (int i = 0; i < N; i++)
        s += h[i];
    return (float)s;
}

// =========================================
// Runner
// =========================================
float run_kernel(void(*kernel)(const float*, float*, int),
                 const float* d_in, float* d_out,
                 int N, int blocks, int threads,
                 size_t smemBytes, int iters)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < 5; ++i)
        kernel<<<blocks, threads, smemBytes>>>(d_in, d_out, N);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        kernel<<<blocks, threads, smemBytes>>>(d_in, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}

// =========================================
// main
// =========================================
int main(int argc, char** argv)
{
    int N = 1 << 20;   // 1M elements
    int threads = 1024;
    int blocks  = (N + threads - 1) / threads;
    int iters   = 100;

    printf("N=%d, blocks=%d, threads=%d\n", N, blocks, threads);

    float* h_in = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    float ref = cpu_reduce(h_in, N);
    printf("CPU ref = %.2f\n", ref);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, blocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    size_t smem_naive = threads * sizeof(float);            // naive smem
    size_t smem_warp  = (threads / 32) * sizeof(float);     // warpSums

    float t_naive = run_kernel(reduce_naive, d_in, d_out,
                               N, blocks, threads,
                               smem_naive, iters);

    float t_warp = run_kernel(reduce_warp, d_in, d_out,
                              N, blocks, threads,
                              smem_warp, iters);

    float* h_out = (float*)malloc(blocks * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_out, d_out,
                          blocks * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float gpu_sum = cpu_reduce(h_out, blocks);
    printf("[naive ] GPU sum = %.2f (diff=%.6f)\n", gpu_sum, gpu_sum - ref);

    CUDA_CHECK(cudaMemcpy(h_out, d_out,
                          blocks * sizeof(float),
                          cudaMemcpyDeviceToHost));
    gpu_sum = cpu_reduce(h_out, blocks);
    printf("[warp  ] GPU sum = %.2f (diff=%.6f)\n", gpu_sum, gpu_sum - ref);

    printf("\n=== Timing ===\n");
    printf("naive reduction : %.4f ms\n", t_naive);
    printf("warp reduction  : %.4f ms\n", t_warp);
    printf("Speedup (naive / warp): %.1fx\n", t_naive / t_warp);

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
/*
nvcc -O3 -arch=sm_86 tree_reduction_test.cu -o tree_reduction_test
.\tree_reduction_test.exe

ncu --kernel-name regex:reduce_naive.*     --set full --launch-skip 5 --launch-count 1 ./tree_reduction_test

ncu --kernel-name regex:reduce_warp.*     --set full --launch-skip 5 --launch-count 1 ./tree_reduction_test

*/