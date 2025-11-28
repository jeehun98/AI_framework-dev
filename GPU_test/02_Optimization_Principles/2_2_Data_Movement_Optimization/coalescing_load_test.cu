#include <cstdio>
#include <cstdlib>
#include <cmath>
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
// 설정값
// =======================
constexpr int THREADS_PER_BLOCK = 256;      // 8 warps per block
constexpr int BLOCKS            = 80;       // 총 warp 수 = 80 * 8 = 640
constexpr int WARP_SIZE         = 32;
constexpr int WARPS_PER_BLOCK   = THREADS_PER_BLOCK / WARP_SIZE;
constexpr int TOTAL_WARPS       = BLOCKS * WARPS_PER_BLOCK;

// 반복 횟수 (load 수 키우기)
constexpr int ITERS             = 1024;

// stride 설정
constexpr int STRIDE            = 32;       // 의도적으로 최악의 coalescing 패턴

// input buffer 크기 (strided 커널 기준으로 잡고, coalesced도 같이 사용)
constexpr size_t TOTAL_ELEMS_STRIDED =
    static_cast<size_t>(TOTAL_WARPS) * ITERS * WARP_SIZE * STRIDE;

// =======================
// 유틸 함수
// =======================
void init_input(float* buf, size_t n, float scale = 1.0f) {
    for (size_t i = 0; i < n; ++i) {
        buf[i] = scale * static_cast<float>(i % 1024) * 0.001f;
    }
}

// =======================
// Coalesced Load 커널
//  - warp 내에서 lane별로 연속된 index 접근
//  - idx_coal = warp_base + it*32 + lane
// =======================
__global__ void coalesced_load_kernel(const float* __restrict__ in,
                                      float* __restrict__ out,
                                      int iters)
{
    int tid      = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id  = tid / 32;           // global warp id
    int lane     = threadIdx.x & 31;   // lane id in warp [0..31]

    // warp당 시작 offset (coalesced는 STRIDE 없이 compact 영역만 사용)
    size_t warp_base = static_cast<size_t>(warp_id) * iters * 32;

    float acc = 0.0f;

    for (int it = 0; it < iters; ++it) {
        size_t idx = warp_base + it * 32 + lane;   // lane끼리 연속 (coalesced)
        float v = in[idx];
        acc += v;
    }

    // 결과 저장 (최소한의 store, 컴파일러 최적화 방지용)
    out[tid] = acc;
}

// =======================
// Strided Load 커널
//  - warp 내에서 lane별로 STRIDE 간격 접근
//  - idx_str = warp_base + (it*32 + lane) * STRIDE
// =======================
__global__ void strided_load_kernel(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int iters,
                                    int stride)
{
    int tid      = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id  = tid / 32;           // global warp id
    int lane     = threadIdx.x & 31;   // lane id in warp [0..31]

    // strided용 warp base (coalesced보다 STRIDE배 넓게 잡힘)
    size_t warp_base = static_cast<size_t>(warp_id) * iters * 32 * stride;

    float acc = 0.0f;

    for (int it = 0; it < iters; ++it) {
        size_t idx_in_warp = static_cast<size_t>(it) * 32 + lane;
        size_t idx         = warp_base + idx_in_warp * stride;
        float v = in[idx];     // lane마다 STRIDE 간격으로 떨어짐 (비-coalesced)
        acc += v;
    }

    out[tid] = acc;
}

// =======================
// main: Coalesced vs Strided 비교
// =======================
int main()
{
    printf("=== Coalescing Test 1: Coalesced vs Strided Load ===\n");
    printf("THREADS_PER_BLOCK = %d, BLOCKS = %d, TOTAL_WARPS = %d\n",
           THREADS_PER_BLOCK, BLOCKS, TOTAL_WARPS);
    printf("ITERS = %d, STRIDE = %d\n\n", ITERS, STRIDE);

    size_t bytes_in  = TOTAL_ELEMS_STRIDED * sizeof(float);
    size_t total_threads = static_cast<size_t>(THREADS_PER_BLOCK) * BLOCKS;
    size_t bytes_out = total_threads * sizeof(float);

    float* h_in  = (float*)malloc(bytes_in);
    float* h_out = (float*)malloc(bytes_out);

    init_input(h_in, TOTAL_ELEMS_STRIDED, 0.01f);

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in,  bytes_in));
    CHECK_CUDA(cudaMalloc(&d_out, bytes_out));

    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes_in, cudaMemcpyHostToDevice));

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCKS);

    // -----------------------
    // 1) Coalesced Load
    // -----------------------
    {
        printf("[coalesced] warm-up + timing\n");
        coalesced_load_kernel<<<grid, block>>>(d_in, d_out, ITERS);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        coalesced_load_kernel<<<grid, block>>>(d_in, d_out, ITERS);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

        printf("[coalesced] kernel time: %.3f ms\n\n", ms);
    }

    // -----------------------
    // 2) Strided Load
    // -----------------------
    {
        printf("[strided] warm-up + timing\n");
        strided_load_kernel<<<grid, block>>>(d_in, d_out, ITERS, STRIDE);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        strided_load_kernel<<<grid, block>>>(d_in, d_out, ITERS, STRIDE);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

        printf("[strided]  kernel time: %.3f ms\n\n", ms);
    }

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    free(h_in);
    free(h_out);

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}
/*
nvcc -O3 -arch=sm_86 -lineinfo -o coalescing_load_test.exe coalescing_load_test.cu

ncu --kernel-name regex:coalesced_load_kernel.* --metrics dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed   ./coalescing_load_test.exe
ncu --kernel-name regex:strided_load_kernel.* --metrics dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed   ./coalescing_load_test.exe


*/