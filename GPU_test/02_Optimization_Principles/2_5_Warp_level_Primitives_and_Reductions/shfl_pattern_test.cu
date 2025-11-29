#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)
#endif

// ========================================
// Warp-level reduction helpers
// ========================================
__inline__ __device__
float warpReduceDown(float val)
{
    unsigned mask = 0xffffffff;
    // tree style: 매 단계에서 상위 lane의 값을 아래 lane으로 내려보냄
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val; // lane 0 에 최종 sum
}

__inline__ __device__
float warpReduceXor(float val)
{
    unsigned mask = 0xffffffff;
    // butterfly / pairwise style: lane ^ offset 와 교환
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(mask, val, offset);
    }
    return val; // 모든 lane 이 같은 값(최종 sum)을 갖게 됨
}

// ========================================
// Block-level reduction kernels
// (warp 내부 패턴만 다르고 구조는 동일)
// ========================================
__global__ void reduce_shfl_down_kernel(const float* __restrict__ g_in,
                                        float* __restrict__ g_out,
                                        int N)
{
    extern __shared__ float warpSums[]; // warp당 1개

    unsigned int tid       = threadIdx.x;
    unsigned int blockSize = blockDim.x;
    unsigned int idx       = blockIdx.x * (blockSize * 2) + tid;

    float val = 0.0f;
    if (idx < N) {
        val += g_in[idx];
    }
    if (idx + blockSize < N) {
        val += g_in[idx + blockSize];
    }

    int lane   = tid & 31; // 0~31
    int warpId = tid >> 5; // warp index in block

    // 1단계: warp 내부 shfl_down reduction
    val = warpReduceDown(val);

    if (lane == 0) {
        warpSums[warpId] = val;
    }
    __syncthreads();

    // 2단계: 첫 warp가 warpSums[] reduction
    float warpVal = (tid < (blockSize / 32)) ? warpSums[lane] : 0.0f;

    if (warpId == 0) {
        float blockSum = warpReduceDown(warpVal);
        if (lane == 0) {
            g_out[blockIdx.x] = blockSum;
        }
    }
}

__global__ void reduce_shfl_xor_kernel(const float* __restrict__ g_in,
                                       float* __restrict__ g_out,
                                       int N)
{
    extern __shared__ float warpSums[];

    unsigned int tid       = threadIdx.x;
    unsigned int blockSize = blockDim.x;
    unsigned int idx       = blockIdx.x * (blockSize * 2) + tid;

    float val = 0.0f;
    if (idx < N) {
        val += g_in[idx];
    }
    if (idx + blockSize < N) {
        val += g_in[idx + blockSize];
    }

    int lane   = tid & 31;
    int warpId = tid >> 5;

    // 1단계: warp 내부 shfl_xor reduction
    val = warpReduceXor(val);

    if (lane == 0) {
        warpSums[warpId] = val;
    }
    __syncthreads();

    // 2단계: 첫 warp가 warpSums[] reduction (xor 패턴 사용)
    float warpVal = (tid < (blockSize / 32)) ? warpSums[lane] : 0.0f;

    if (warpId == 0) {
        float blockSum = warpReduceXor(warpVal);
        if (lane == 0) {
            g_out[blockIdx.x] = blockSum;
        }
    }
}

// ========================================
// Host helpers
// ========================================
float cpu_reduce(const float* h, int N)
{
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += h[i];
    }
    return static_cast<float>(sum);
}

void run_kernel(const char* name,
                void(*kernel)(const float*, float*, int),
                const float* d_in,
                float* d_blockSums,
                int N,
                int blocks,
                int threads,
                size_t smemBytes,
                int iters,
                float& out_ms)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warm-up
    for (int i = 0; i < 5; ++i) {
        kernel<<<blocks, threads, smemBytes>>>(d_in, d_blockSums, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        kernel<<<blocks, threads, smemBytes>>>(d_in, d_blockSums, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&out_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int argc, char** argv)
{
    const int N = 1 << 24; // ~16M
    const int threads = 256;
    const int blocks  = (N + threads * 2 - 1) / (threads * 2);

    int iters = 100;
    if (argc > 1) {
        iters = std::atoi(argv[1]);
        if (iters <= 0) iters = 1;
    }

    printf("N = %d, blocks = %d, threads = %d, iters = %d\n",
           N, blocks, threads, iters);

    // Host input
    float* h_in = (float*)std::malloc(sizeof(float) * N);
    if (!h_in) {
        fprintf(stderr, "host alloc failed\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < N; ++i) {
        h_in[i] = 1.0f;
    }

    float ref = cpu_reduce(h_in, N);
    printf("CPU reference sum = %.2f\n", ref);

    // Device buffers
    float *d_in = nullptr, *d_blockSums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&d_blockSums, sizeof(float) * blocks));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeof(float) * N, cudaMemcpyHostToDevice));

    size_t smemBytes = (threads / 32) * sizeof(float); // warp 수만큼 shared

    float ms_down = 0.0f;
    float ms_xor  = 0.0f;

    float* h_blockSums = (float*)std::malloc(sizeof(float) * blocks);

    // shfl_down 버전
    run_kernel("reduce_shfl_down_kernel",
               (void(*)(const float*, float*, int))reduce_shfl_down_kernel,
               d_in, d_blockSums, N, blocks, threads, smemBytes, iters, ms_down);

    CUDA_CHECK(cudaMemcpy(h_blockSums, d_blockSums,
                          sizeof(float) * blocks, cudaMemcpyDeviceToHost));
    float gpu_sum_down = cpu_reduce(h_blockSums, blocks);
    printf("[shfl_down] GPU sum = %.2f (diff = %.6f)\n",
           gpu_sum_down, gpu_sum_down - ref);

    // shfl_xor 버전
    run_kernel("reduce_shfl_xor_kernel",
               (void(*)(const float*, float*, int))reduce_shfl_xor_kernel,
               d_in, d_blockSums, N, blocks, threads, smemBytes, iters, ms_xor);

    CUDA_CHECK(cudaMemcpy(h_blockSums, d_blockSums,
                          sizeof(float) * blocks, cudaMemcpyDeviceToHost));
    float gpu_sum_xor = cpu_reduce(h_blockSums, blocks);
    printf("[shfl_xor ] GPU sum = %.2f (diff = %.6f)\n",
           gpu_sum_xor, gpu_sum_xor - ref);

    printf("\n=== Timing (averaged over %d iterations) ===\n", iters);
    printf("shfl_down reduction : %.3f ms\n", ms_down / iters);
    printf("shfl_xor  reduction : %.3f ms\n", ms_xor  / iters);
    printf("Speedup (down / xor): %.2fx\n", (ms_down / ms_xor));

    std::free(h_in);
    std::free(h_blockSums);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_blockSums));

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
/*
nvcc -O3 -arch=sm_86 shfl_pattern_test.cu -o shfl_pattern_test
./shfl_pattern_test        # 기본 iters=100
./shfl_pattern_test 10     # 빠르게 테스트 하고 싶을 때

# shfl_down 패턴
ncu   --kernel-name regex:reduce_shfl_down_kernel.*   --launch-skip 5   --launch-count 1   --set full   ./shfl_pattern_test 20

# shfl_xor 패턴
ncu   --kernel-name regex:reduce_shfl_xor_kernel.*   --launch-skip 5   --launch-count 1   --set full   ./shfl_pattern_test 20

ncu ^
  --kernel-name regex:reduce_shfl_xor_kernel.* ^
  --launch-skip 5 ^
  --launch-count 1 ^
  --set full ^
  ./shfl_pattern_test 20

*/