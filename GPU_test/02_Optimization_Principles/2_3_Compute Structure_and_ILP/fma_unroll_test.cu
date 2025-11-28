#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error %s:%d: %s\n",                           \
                    __FILE__, __LINE__, cudaGetErrorString(err__));             \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)
#endif

// =======================
// 설정
// =======================

constexpr int THREADS_PER_BLOCK = 256;
constexpr int BLOCKS            = 80;
constexpr int TOTAL_THREADS     = THREADS_PER_BLOCK * BLOCKS;

// 각 thread가 수행할 총 FMA 개수 (모든 UNROLL에서 동일하게 유지)
constexpr int TOTAL_FMA_PER_THREAD = 1 << 20;  // 1M FMA/thread 정도

// =======================
// kernel: loop unrolling 실험
// =======================
//
// - template<int UNROLL>
// - 각 thread는 UNROLL 개의 독립 accumulator를 사용
// - 바깥 루프 횟수 = TOTAL_FMA_PER_THREAD / UNROLL
//   → 모든 UNROLL에서 총 FMA 수는 동일
// - UNROLL 증가:
//   - ILP 증가 (동시에 날릴 수 있는 FMA 수 ↑)
//   - register 수 증가 → occupancy 하락 가능
//

template<int UNROLL>
__global__ void fma_unroll_kernel(float* __restrict__ out,
                                  int iters_per_thread)
{
    static_assert(UNROLL >= 1, "UNROLL must be >= 1");

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= TOTAL_THREADS) return;

    // UNROLL 개수만큼 독립 accumulator (ILP용)
    float acc[UNROLL];

    // 초기값 다양하게 (thread ID 섞기)
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        acc[i] = (tid % 17) * 0.1f + i * 0.01f;
    }

    const float a = 1.000001f;
    const float b = 0.000001f;

    // 메인 루프
    for (int it = 0; it < iters_per_thread; ++it) {
#pragma unroll UNROLL
        for (int j = 0; j < UNROLL; ++j) {
            // 서로 독립적인 FMA → ILP 증가
            acc[j] = acc[j] * a + b;
        }
    }

    // 결과 합쳐서 기록 (dead-code 제거 방지)
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < UNROLL; ++j) {
        sum += acc[j];
    }
    out[tid] = sum;
}

// =======================
// host util
// =======================

template<int UNROLL>
void run_unroll_config(float* d_out)
{
    const int iters_per_thread = TOTAL_FMA_PER_THREAD / UNROLL;

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCKS);

    // warm-up
    fma_unroll_kernel<UNROLL><<<grid, block>>>(d_out, iters_per_thread);
    CHECK_CUDA(cudaDeviceSynchronize());

    // timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    fma_unroll_kernel<UNROLL><<<grid, block>>>(d_out, iters_per_thread);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    // 대충 결과 하나만 읽어서 side-effect 보장
    float h0;
    CHECK_CUDA(cudaMemcpy(&h0, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    printf("Config UNROLL=%d  |  iters/thread=%d  |  time=%.3f ms  |  sample out=%.6f\n",
           UNROLL, iters_per_thread, ms, h0);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// =======================
// main
// =======================

int main()
{
    printf("=== Loop Unrolling Test: UNROLL = 1,2,4,8 ===\n");
    printf("Threads: %d blocks x %d threads = %d threads\n",
           BLOCKS, THREADS_PER_BLOCK, TOTAL_THREADS);
    printf("TOTAL_FMA_PER_THREAD = %d (all configs)\n\n", TOTAL_FMA_PER_THREAD);

    float* d_out;
    CHECK_CUDA(cudaMalloc(&d_out, TOTAL_THREADS * sizeof(float)));

    // UNROLL sweep
    run_unroll_config<1>(d_out);
    run_unroll_config<2>(d_out);
    run_unroll_config<4>(d_out);
    run_unroll_config<8>(d_out);

    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaDeviceReset());
    return 0;
}

/*

nvcc -O3 -arch=sm_86 -lineinfo -o fma_unroll_test.exe fma_unroll_test.cu

ncu --kernel-name regex:fma_unroll_kernel.* --metrics smsp__inst_executed.sum,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed   ./fma_unroll_test.exe

*/