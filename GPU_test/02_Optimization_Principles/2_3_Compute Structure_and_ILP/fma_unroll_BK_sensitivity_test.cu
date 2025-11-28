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

// =======================================
// 설정
// =======================================

constexpr int THREADS_PER_BLOCK = 256;
constexpr int BLOCKS            = 80;
constexpr int TOTAL_THREADS     = THREADS_PER_BLOCK * BLOCKS;

// 각 thread가 처리하는 "타일 반복" 수
//   FMA/thread = BK * TILES_PER_THREAD
//   BK가 커질수록 연산량도 함께 증가 (BK 민감도 보기용)
constexpr int TILES_PER_THREAD  = 2048;

// =======================================
// kernel: BK × UNROLL 민감도 실험
// =======================================
//
// template<int BK, int UNROLL>
//
// - BK: inner K-loop 길이
// - UNROLL: K-loop 내의 "독립 accumulator" 개수
//   (UNROLL 개의 acc를 써서 ILP를 키움)
// - BK는 UNROLL의 배수라고 가정 (BK % UNROLL == 0)
//
// 구조:
//
// for tile in 0..TILES_PER_THREAD-1:
//   for k in 0..(BK/UNROLL - 1):
//     for u in 0..UNROLL-1:
//       acc[u] = acc[u] * a + b;
//
// → tile당 FMA/thread = BK
// → 전체 FMA/thread = BK * TILES_PER_THREAD
//

template<int BK, int UNROLL>
__global__ void fma_unroll_BK_kernel(float* __restrict__ out,
                                     int tiles_per_thread)
{
    static_assert(BK > 0, "BK must be > 0");
    static_assert(UNROLL >= 1, "UNROLL must be >= 1");
    static_assert(BK % UNROLL == 0, "BK must be divisible by UNROLL");

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= TOTAL_THREADS) return;

    // UNROLL 개수만큼 독립 accumulator 사용 (ILP 확보)
    float acc[UNROLL];

#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
        acc[u] = (tid % 31) * 0.01f + 0.001f * u;
    }

    const float a = 1.000001f;
    const float b = 0.000001f;

    const int inner_iters = BK / UNROLL;

    // 메인 루프: TILES_PER_THREAD 동안 반복
    for (int t = 0; t < tiles_per_thread; ++t) {
#pragma unroll
        for (int ik = 0; ik < inner_iters; ++ik) {
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                // 독립적인 UNROLL 개수만큼의 FMA
                acc[u] = acc[u] * a + b;
            }
        }
    }

    // 결과 합산해서 기록 (dead-code 제거 방지)
    float sum = 0.0f;
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
        sum += acc[u];
    }
    out[tid] = sum;
}

// =======================================
// host helper
// =======================================

template<int BK, int UNROLL>
void run_BK_unroll_config(float* d_out)
{
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCKS);

    const int tiles_per_thread = TILES_PER_THREAD;
    const long long fma_per_thread = 1LL * BK * tiles_per_thread;

    // warm-up
    fma_unroll_BK_kernel<BK, UNROLL><<<grid, block>>>(d_out, tiles_per_thread);
    CHECK_CUDA(cudaDeviceSynchronize());

    // timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    fma_unroll_BK_kernel<BK, UNROLL><<<grid, block>>>(d_out, tiles_per_thread);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    float sample = 0.0f;
    CHECK_CUDA(cudaMemcpy(&sample, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    printf("Config BK=%2d, UNROLL=%d  |  tiles/thread=%d, FMA/thread=%lld  |  time=%.3f ms  |  sample=%.6f\n",
           BK, UNROLL, tiles_per_thread, fma_per_thread, ms, sample);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// =======================================
// main
// =======================================

int main()
{
    printf("=== Loop Unrolling Test 2: BK sensitivity (BK=4,8,16,32 × UNROLL=1,2,4,8) ===\n");
    printf("Threads: %d blocks x %d threads = %d threads\n",
           BLOCKS, THREADS_PER_BLOCK, TOTAL_THREADS);
    printf("TILES_PER_THREAD = %d\n", TILES_PER_THREAD);
    printf("FMA/thread = BK * TILES_PER_THREAD (UNROLL과 무관)\n\n");

    float* d_out;
    CHECK_CUDA(cudaMalloc(&d_out, TOTAL_THREADS * sizeof(float)));

    // BK = 4
    run_BK_unroll_config<4,1>(d_out);
    run_BK_unroll_config<4,2>(d_out);
    run_BK_unroll_config<4,4>(d_out);
    printf("\n");

    // BK = 8
    run_BK_unroll_config<8,1>(d_out);
    run_BK_unroll_config<8,2>(d_out);
    run_BK_unroll_config<8,4>(d_out);
    run_BK_unroll_config<8,8>(d_out);
    printf("\n");

    // BK = 16
    run_BK_unroll_config<16,1>(d_out);
    run_BK_unroll_config<16,2>(d_out);
    run_BK_unroll_config<16,4>(d_out);
    run_BK_unroll_config<16,8>(d_out);
    printf("\n");

    // BK = 32
    run_BK_unroll_config<32,1>(d_out);
    run_BK_unroll_config<32,2>(d_out);
    run_BK_unroll_config<32,4>(d_out);
    run_BK_unroll_config<32,8>(d_out);
    printf("\n");

    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaDeviceReset());
    return 0;
}
/*
nvcc -O3 -arch=sm_86 -lineinfo -o fma_unroll_BK_sensitivity_test.exe fma_unroll_BK_sensitivity_test.cu

ncu --kernel-name regex:fma_unroll_BK_kernel.* --metrics smsp__inst_executed.sum,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed   ./fma_unroll_BK_sensitivity_test.exe

*/