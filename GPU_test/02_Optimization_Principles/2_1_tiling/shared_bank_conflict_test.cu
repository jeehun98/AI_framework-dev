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
constexpr int WARP_SIZE    = 32;
constexpr int THREADS_PER_BLOCK = 256;     // 8 warps
constexpr int BLOCKS             = 80;     // GPU에 맞게 조절 가능
constexpr int ITERS              = 100000; // conflict 효과를 키우기 위한 반복

// =======================
// 1) Bank conflict 유발 kernel
//    - shared[32][32] : column=0 고정 접근 -> 32-way conflict
// =======================
__global__ void bank_conflict_kernel(float* __restrict__ out)
{
    // row-major: sh[row][col]
    __shared__ float sh[WARP_SIZE][WARP_SIZE];

    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;  // 0~31

    // 초기화: 각 row의 col=0에 lane 값 저장
    if (lane < WARP_SIZE) {
        sh[lane][0] = static_cast<float>(lane);
    }
    __syncthreads();

    float acc = 0.0f;

    // 각 warp의 lane들이 모두 sh[lane][0]을 읽음
    // sh[lane][0] : index = lane * 32 + 0 -> bank = (index) % 32 = 0
    // => 같은 warp 내 32 thread가 전부 bank 0을 두드림 (32-way conflict)
    #pragma unroll 4
    for (int it = 0; it < ITERS; ++it) {
        float v = sh[lane][0];
        acc += v * 1.0000001f;  // 약간 비선형, 최적화 방지용
    }

    // 결과를 global에 저장해 최적화 방지
    out[tid] = acc;
}

// =======================
// 2) Padding(+1)으로 bank conflict 제거 kernel
//    - shared[32][33] : column=0 접근 -> no conflict
// =======================
__global__ void bank_conflict_padded_kernel(float* __restrict__ out)
{
    // row-major: sh[row][col]
    // 2D 배열이 [32][33]이면, sh[row][0] index = row*33
    // => bank = (row*33) % 32 = row (모든 lane이 서로 다른 bank)
    __shared__ float sh[WARP_SIZE][WARP_SIZE + 1];

    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;  // 0~31

    if (lane < WARP_SIZE) {
        sh[lane][0] = static_cast<float>(lane);
    }
    __syncthreads();

    float acc = 0.0f;

    #pragma unroll 4
    for (int it = 0; it < ITERS; ++it) {
        float v = sh[lane][0];  // 이제 lane별로 bank 분산
        acc += v * 1.0000001f;
    }

    out[tid] = acc;
}

// =======================
// main: 두 kernel 시간 비교
// =======================
int main()
{
    printf("=== Test 3: Shared Memory Bank Conflict ===\n");
    printf("THREADS_PER_BLOCK = %d, BLOCKS = %d, ITERS = %d\n",
           THREADS_PER_BLOCK, BLOCKS, ITERS);
    printf("bank_conflict_kernel:  sh[32][32]  ->  column=0 access (32-way conflict)\n");
    printf("padded_kernel:        sh[32][33]  ->  column=0 access (no conflict)\n\n");

    const int num_threads = THREADS_PER_BLOCK * BLOCKS;
    size_t bytes = sizeof(float) * num_threads;

    float* d_out;
    CHECK_CUDA(cudaMalloc(&d_out, bytes));

    // -----------------------
    // 1) conflict kernel
    // -----------------------
    {
        printf("[conflict] warm-up + timing\n");
        bank_conflict_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_out);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        bank_conflict_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_out);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        printf("[conflict] kernel time: %.3f ms\n\n", ms);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    // -----------------------
    // 2) padded kernel
    // -----------------------
    {
        printf("[padded] warm-up + timing\n");
        bank_conflict_padded_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_out);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        bank_conflict_padded_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_out);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        printf("[padded]  kernel time: %.3f ms\n\n", ms);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaDeviceReset());
    return 0;
}
// nvcc -O3 -arch=sm_86 -lineinfo -o shared_bank_conflict_test.exe shared_bank_conflict_test.cu

/*
ncu --kernel-name regex:bank_conflict_kernel.* --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__warp_issue_stalled_lg_throttle_per_warp_active,smsp__warp_issue_stalled_long_scoreboard_per_warp_active ./shared_bank_conflict_test.exe

ncu --kernel-name regex:bank_conflict_padded_kernel.* --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__warp_issue_stalled_lg_throttle_per_warp_active,smsp__warp_issue_stalled_long_scoreboard_per_warp_active ./shared_bank_conflict_test.exe
*/