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
constexpr int THREADS_PER_BLOCK = 32;    // 1 warp per block
constexpr int BLOCKS            = 80;    // 80 warps
constexpr int TILES_PER_WARP    = 512;   // warp가 순서대로 읽을 tile 개수
                                         // (BK loop 길이와 유사한 역할)

// 전체 데이터: block(=warp)마다 TILES_PER_WARP * 32 floats
constexpr int TOTAL_FLOATS      = BLOCKS * TILES_PER_WARP * WARP_SIZE;

// =======================
// cp.async 유틸
// =======================

__device__ __forceinline__ void cp_async_4B(void* smem_ptr, const void* gmem_ptr) {
#if __CUDA_ARCH__ >= 800
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(smem_addr), "l"(gmem_ptr)
    );
#else
    // fallback: 그냥 동기 로드
    *reinterpret_cast<float*>(smem_ptr) = *reinterpret_cast<const float*>(gmem_ptr);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template<int STAGES>
__device__ __forceinline__ void cp_async_wait_stage() {
#if __CUDA_ARCH__ >= 800
    // 최대 STAGES개의 group이 outstanding 되도록 허용:
    // → wait_group(STAGES-1)는 "최소 한 group은 완료된 상태"를 보장
    asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 1));
#endif
}

// =======================
// 다단계 cp.async pipeline kernel
//   - STAGES = 2 or 3
//   - warp 당 TILES_PER_WARP chunk 처리
//   - 각 chunk = 32 floats
// =======================

template<int STAGES>
__global__ void cp_async_pipeline_kernel(const float* __restrict__ g_in,
                                         float* __restrict__ g_out,
                                         int tiles_per_warp)
{
    static_assert(STAGES >= 2 && STAGES <= 4, "STAGES must be 2~4");

    __shared__ float sh[STAGES][WARP_SIZE];   // stage별 warp-local buffer

    int lane   = threadIdx.x;   // 0..31
    int warp_id = blockIdx.x;   // block당 1 warp

    // 이 warp가 처리할 global base offset
    int warp_base = warp_id * tiles_per_warp * WARP_SIZE;

    float acc = 0.0f;

    // -------------------------
    // Prologue: 처음 STAGES개 tile prefetch
    // -------------------------
    int prefetched = 0;
    int initial_tiles = (tiles_per_warp < STAGES) ? tiles_per_warp : STAGES;

    for (int t = 0; t < initial_tiles; ++t) {
        int stage = t % STAGES;

        int tile_offset = warp_base + t * WARP_SIZE + lane;
        const float* src = g_in + tile_offset;
        float* dst       = &sh[stage][lane];

        cp_async_4B(dst, src);
        cp_async_commit();
        ++prefetched;
    }

    // 최소 한 tile은 사용 가능하도록 기다림
    if (initial_tiles > 0) {
        cp_async_wait_stage<STAGES>();
    }

    // -------------------------
    // Main loop:
    //   consumed: 지금까지 소비한 tile 개수
    //   prefetched: gmem에서 cp.async 발행한 tile 개수
    // -------------------------
    int consumed = 0;

    while (consumed < tiles_per_warp) {
        int stage = consumed % STAGES;

        // 현재 stage의 데이터 소비
        float v = sh[stage][lane];
        // 약간의 연산 (실제 GEMM이라 가정하면 FMA로 채워지는 부분)
        acc = acc * 1.0001f + v;

        // 다음 tile prefetch (가능하면)
        if (prefetched < tiles_per_warp) {
            int stage_next = prefetched % STAGES;

            int tile_offset = warp_base + prefetched * WARP_SIZE + lane;
            const float* src = g_in + tile_offset;
            float* dst       = &sh[stage_next][lane];

            cp_async_4B(dst, src);
            cp_async_commit();
            ++prefetched;
        }

        // 다음 iteration에서 사용할 tile이 준비되도록 기다림
        if (consumed + 1 < tiles_per_warp) {
            cp_async_wait_stage<STAGES>();
        }

        ++consumed;
    }

    // warp별 결과 저장 (lane 단위)
    int out_idx = warp_id * WARP_SIZE + lane;
    g_out[out_idx] = acc;
}

// =======================
// host util
// =======================

void init_data(float* a, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] = (i % 17) * 0.1f;
    }
}

// =======================
// main
// =======================

int main()
{
    printf("=== cp.async Test 2: 2-stage vs 3-stage pipeline ===\n");
    printf("THREADS_PER_BLOCK = %d, BLOCKS = %d\n", THREADS_PER_BLOCK, BLOCKS);
    printf("TILES_PER_WARP = %d (BK loop 길이와 유사)\n", TILES_PER_WARP);
    printf("Total floats = %d\n\n", TOTAL_FLOATS);

    size_t bytes = TOTAL_FLOATS * sizeof(float);

    float* h_in  = (float*)malloc(bytes);
    float* h_out = (float*)malloc(BLOCKS * WARP_SIZE * sizeof(float));

    init_data(h_in, TOTAL_FLOATS);

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, BLOCKS * WARP_SIZE * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCKS);

    // -------------------------
    // 2-stage pipeline (STAGES=2)
    // -------------------------
    {
        printf("[2-stage] warm-up + timing\n");
        cp_async_pipeline_kernel<2><<<grid, block>>>(d_in, d_out, TILES_PER_WARP);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        cp_async_pipeline_kernel<2><<<grid, block>>>(d_in, d_out, TILES_PER_WARP);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        CHECK_CUDA(cudaMemcpy(h_out, d_out,
                              BLOCKS * WARP_SIZE * sizeof(float),
                              cudaMemcpyDeviceToHost));

        printf("[2-stage] kernel time: %.3f ms\n\n", ms);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    // -------------------------
    // 3-stage pipeline (STAGES=3)
    // -------------------------
    {
        printf("[3-stage] warm-up + timing\n");
        cp_async_pipeline_kernel<3><<<grid, block>>>(d_in, d_out, TILES_PER_WARP);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        cp_async_pipeline_kernel<3><<<grid, block>>>(d_in, d_out, TILES_PER_WARP);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        CHECK_CUDA(cudaMemcpy(h_out, d_out,
                              BLOCKS * WARP_SIZE * sizeof(float),
                              cudaMemcpyDeviceToHost));

        printf("[3-stage] kernel time: %.3f ms\n\n", ms);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    free(h_in);
    free(h_out);

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}
/*
nvcc -O3 -arch=sm_86 -lineinfo -o cp_async_pipeline_stages_test.exe cp_async_pipeline_stages_test.cu

ncu --kernel-name regex:cp_async_pipeline_kernel.* --metrics smsp__warp_issue_stalled_memory_dependency_per_warp_active.avg,smsp__warp_issue_stalled_membar_per_warp_active.avg,smsp__pipe_lsu_cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed   ./cp_async_pipeline_stages_test.exe

ncu ^
  --kernel-name regex:cp_async_pipeline_kernel<3>.* ^
  --metrics \
smsp__warp_issue_stalled_memory_dependency_per_warp_active.avg,\
smsp__warp_issue_stalled_membar_per_warp_active.avg,\
smsp__pipe_lsu_cycles_active.avg.pct_of_peak_sustained_elapsed,\
smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed \
  ./cp_async_pipeline_stages_test.exe

*/