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

constexpr int WARP_SIZE         = 32;
constexpr int THREADS_PER_BLOCK = 32;   // 1 warp/block
constexpr int BLOCKS            = 80;   // 총 80 warps
constexpr int ITERS             = 100000;

constexpr int STRIDE            = 32;   // conflict stride
constexpr int STRIDE_PAD        = STRIDE + 1;

// shared 배열 크기: lane * STRIDE_PAD까지 커버
constexpr int SH_SIZE           = WARP_SIZE * STRIDE_PAD;  // 32 * 33 = 1056

// =======================
// conflict kernel
//   sh[lane * STRIDE]  (STRIDE=32 → 모두 같은 bank → 32-way conflict)
// =======================
__global__ void bank_conflict_stride_kernel(float* __restrict__ out)
{
    __shared__ float sh[SH_SIZE];

    int tid  = threadIdx.x;          // 0..31
    int lane = tid;                  // 단일 warp

    // 초기화: 각 lane이 자신이 접근할 위치에 값 하나 저장
    int idx_conflict = lane * STRIDE;
    if (idx_conflict < SH_SIZE) {
        sh[idx_conflict] = static_cast<float>(lane);
    }
    __syncthreads();

    float acc = 0.0f;

    // 동일한 주소 패턴으로 ITERS번 load
    for (int it = 0; it < ITERS; ++it) {
        float v = sh[idx_conflict];
        acc += v;
    }

    // 결과를 global에 기록해서 최적화 방지
    int global_idx = blockIdx.x * blockDim.x + lane;
    out[global_idx] = acc;
}

// =======================
// padding kernel
//   sh[lane * (STRIDE+1)]
//   → bank conflict 제거
// =======================
__global__ void bank_conflict_padded_stride_kernel(float* __restrict__ out)
{
    __shared__ float sh[SH_SIZE];

    int tid  = threadIdx.x;      // 0..31
    int lane = tid;

    int idx_padded = lane * STRIDE_PAD;
    if (idx_padded < SH_SIZE) {
        sh[idx_padded] = static_cast<float>(lane);
    }
    __syncthreads();

    float acc = 0.0f;

    for (int it = 0; it < ITERS; ++it) {
        float v = sh[idx_padded];
        acc += v;
    }

    int global_idx = blockIdx.x * blockDim.x + lane;
    out[global_idx] = acc;
}

// =======================
// main
// =======================
int main()
{
    printf("=== Test 1: Shared Memory Bank Conflict (stride vs padding) ===\n");
    printf("THREADS_PER_BLOCK = %d, BLOCKS = %d, ITERS = %d\n",
           THREADS_PER_BLOCK, BLOCKS, ITERS);
    printf("STRIDE = %d (conflict), STRIDE_PAD = %d (no conflict)\n",
           STRIDE, STRIDE_PAD);
    printf("SH_SIZE = %d (floats, per block)\n\n", SH_SIZE);

    int total_threads = THREADS_PER_BLOCK * BLOCKS;
    size_t bytes_out  = total_threads * sizeof(float);

    float* d_out;
    float* h_out = (float*)malloc(bytes_out);

    CHECK_CUDA(cudaMalloc(&d_out, bytes_out));

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCKS);

    // -------------------------
    // 1) conflict stride kernel
    // -------------------------
    {
        printf("[conflict_stride] warm-up + timing\n");
        bank_conflict_stride_kernel<<<grid, block>>>(d_out);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        bank_conflict_stride_kernel<<<grid, block>>>(d_out);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes_out, cudaMemcpyDeviceToHost));

        printf("[conflict_stride] kernel time: %.3f ms\n\n", ms);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    // -------------------------
    // 2) padded stride kernel
    // -------------------------
    {
        printf("[padded_stride] warm-up + timing\n");
        bank_conflict_padded_stride_kernel<<<grid, block>>>(d_out);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        bank_conflict_padded_stride_kernel<<<grid, block>>>(d_out);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes_out, cudaMemcpyDeviceToHost));

        printf("[padded_stride]  kernel time: %.3f ms\n\n", ms);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    CHECK_CUDA(cudaFree(d_out));
    free(h_out);

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}

/*
nvcc -O3 -arch=sm_86 -lineinfo -o shared_bank_conflict_padding_test.exe shared_bank_conflict_padding_test.cu


ncu --kernel-name regex:bank_conflict_stride_kernel.* --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,smsp__warp_issue_stalled_bank_conflict_per_warp_active.avg,smsp__warp_issue_stalled_bank_conflict_per_warp_active.pct ./shared_bank_conflict_padding_test.exe
ncu --kernel-name regex:bank_conflict_padded_stride_kernel.* --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,smsp__warp_issue_stalled_bank_conflict_per_warp_active.avg,smsp__warp_issue_stalled_bank_conflict_per_warp_active.pct ./shared_bank_conflict_padding_test.exe

*/