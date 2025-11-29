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

// ----------------------------------------
// Shared-memory 기반 reduction
// ----------------------------------------
__global__ void reduce_shared_kernel(const float* __restrict__ g_in,
                                     float* __restrict__ g_out,
                                     int N)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;
    unsigned int idx = blockIdx.x * (blockSize * 2) + tid;

    float val = 0.0f;
    if (idx < N) {
        val += g_in[idx];
    }
    if (idx + blockSize < N) {
        val += g_in[idx + blockSize];
    }

    sdata[tid] = val;
    __syncthreads();

    // smem tree reduction
    for (unsigned int s = blockSize / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 마지막 warp도 shared memory로 처리 (의도적으로 shuffle 미사용)
    if (tid < 32) {
        // warp-synchronous 영역: __syncthreads() 없이 shared를 여러 번 read
        volatile float* vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    if (tid == 0) {
        g_out[blockIdx.x] = sdata[0];
    }
}

// ----------------------------------------
// Warp shuffle 기반 reduction
// ----------------------------------------
__inline__ __device__
float warpReduceSum(float val)
{
    // 정도만 맞으면 되니 full mask 사용
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__global__ void reduce_shuffle_kernel(const float* __restrict__ g_in,
                                      float* __restrict__ g_out,
                                      int N)
{
    // 각 warp의 partial sum만 shared에 저장 -> shared 트랜잭션 최소화
    extern __shared__ float warpSums[];

    unsigned int tid      = threadIdx.x;
    unsigned int blockSize = blockDim.x;
    unsigned int idx      = blockIdx.x * (blockSize * 2) + tid;

    float val = 0.0f;
    if (idx < N) {
        val += g_in[idx];
    }
    if (idx + blockSize < N) {
        val += g_in[idx + blockSize];
    }

    int lane  = tid & 31;          // warp 내 lane
    int warpId = tid >> 5;         // warp index in block

    // 1단계: warp 내부는 shuffle로 reduction
    val = warpReduceSum(val);

    // warp당 하나씩 shared에 저장
    if (lane == 0) {
        warpSums[warpId] = val;
    }
    __syncthreads();

    // 2단계: 첫 warp가 warpSums[]를 다시 reduction
    float warpVal = (tid < (blockSize / 32)) ? warpSums[lane] : 0.0f;

    if (warpId == 0) {
        float blockSum = warpReduceSum(warpVal);
        if (lane == 0) {
            g_out[blockIdx.x] = blockSum;
        }
    }
}

// ----------------------------------------
// Host 유틸 함수
// ----------------------------------------
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

int main()
{
    const int N = 1 << 24;  // ~16M elements
    const int threads = 256;
    const int blocks = (N + threads * 2 - 1) / (threads * 2);
    const int iters = 1;

    printf("N = %d, blocks = %d, threads = %d\n", N, blocks, threads);

    // Host 메모리
    float* h_in = (float*)malloc(sizeof(float) * N);
    if (!h_in) {
        fprintf(stderr, "host alloc failed\n");
        return EXIT_FAILURE;
    }

    // 입력 초기화 (간단히 1.0f)
    for (int i = 0; i < N; ++i) {
        h_in[i] = 1.0f;
    }

    float ref = cpu_reduce(h_in, N);
    printf("CPU reference sum = %.2f\n", ref);

    // Device 메모리
    float *d_in = nullptr, *d_blockSums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&d_blockSums, sizeof(float) * blocks));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeof(float) * N, cudaMemcpyHostToDevice));

    // Shared memory 크기
    size_t smem_shared  = threads * sizeof(float);
    size_t smem_shuffle = (threads / 32) * sizeof(float); // warp 수 만큼만

    // 커널 실행 및 타이밍
    float ms_shared  = 0.0f;
    float ms_shuffle = 0.0f;

    // 함수 포인터 캐스팅 (nvcc에서 허용)
    run_kernel("reduce_shared_kernel",
               (void(*)(const float*, float*, int))reduce_shared_kernel,
               d_in, d_blockSums, N, blocks, threads, smem_shared, iters, ms_shared);

    // 결과 확인
    float* h_blockSums = (float*)malloc(sizeof(float) * blocks);
    CUDA_CHECK(cudaMemcpy(h_blockSums, d_blockSums, sizeof(float) * blocks, cudaMemcpyDeviceToHost));
    float gpu_sum_shared = cpu_reduce(h_blockSums, blocks);
    printf("[shared ] GPU sum = %.2f (diff = %.6f)\n", gpu_sum_shared, gpu_sum_shared - ref);

    run_kernel("reduce_shuffle_kernel",
               (void(*)(const float*, float*, int))reduce_shuffle_kernel,
               d_in, d_blockSums, N, blocks, threads, smem_shuffle, iters, ms_shuffle);

    CUDA_CHECK(cudaMemcpy(h_blockSums, d_blockSums, sizeof(float) * blocks, cudaMemcpyDeviceToHost));
    float gpu_sum_shuffle = cpu_reduce(h_blockSums, blocks);
    printf("[shuffle] GPU sum = %.2f (diff = %.6f)\n", gpu_sum_shuffle, gpu_sum_shuffle - ref);

    // 평균 kernel 시간 (ms)
    printf("\n=== Timing (averaged over %d iterations) ===\n", iters);
    printf("Shared-memory reduction : %.3f ms\n", ms_shared  / iters);
    printf("Warp-shuffle reduction  : %.3f ms\n", ms_shuffle / iters);
    printf("Speedup (shared / shuffle): %.2fx\n", (ms_shared / ms_shuffle));

    // 정리
    free(h_in);
    free(h_blockSums);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_blockSums));

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
/*

nvcc -O3 -arch=sm_86 reduction_shuffle_test.cu -o reduction_shuffle_test

./reduction_shuffle_test

ncu   --kernel-name regex:reduce_shared_kernel.*   --metrics smsp__warp_cycles_per_stall_barrier_per_warp_active.avg,smsp__warp_cycles_per_stall_memory_dependency_per_warp_active.avg,smsp__inst_executed_shared.sum,l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_shared_op_st.sum   ./reduction_shuffle_test

ncu   --kernel-name regex:reduce_shuffle_kernel.*   --metrics smsp__warp_cycles_per_stall_barrier_per_warp_active.avg,smsp__warp_cycles_per_stall_memory_dependency_per_warp_active.avg,smsp__inst_executed_shared.sum,l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_shared_op_st.sum   ./reduction_shuffle_test


ncu  --kernel-name regex:reduce_shuffle_kernel.*   --set full   ./reduction_shuffle_test

ncu --kernel-name regex:reduce_shared_kernel.*   --launch-skip 5   --launch-count 1   ./reduction_shuffle_test
ncu --kernel-name regex:reduce_shuffle_kernel.*   --launch-skip 5   --launch-count 1   ./reduction_shuffle_test

*/