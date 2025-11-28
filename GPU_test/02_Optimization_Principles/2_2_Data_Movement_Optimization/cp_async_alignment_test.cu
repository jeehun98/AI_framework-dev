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

// warp 하나가 읽는 tile 개수 (BK loop 길이 느낌)
constexpr int TILES_PER_WARP    = 512;

// 전체 float 개수 (misalign 1 float 여유 줌)
constexpr int TOTAL_FLOATS      = BLOCKS * TILES_PER_WARP * WARP_SIZE + 1;

// =======================
// cp.async 유틸
// =======================

__device__ __forceinline__ void cp_async_4B(void* smem_ptr, const void* gmem_ptr)
{
#if __CUDA_ARCH__ >= 800
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr)
    );
#else
    // fallback (pre-SM80): 그냥 동기 로드
    *reinterpret_cast<float*>(smem_ptr) = *reinterpret_cast<const float*>(gmem_ptr);
#endif
}

__device__ __forceinline__ void cp_async_commit()
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait_all()
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

// =======================
// alignment 테스트용 kernel
//   - MISALIGN_FLOATS = 0  → 128B 정렬 (warp당 tile 시작 주소가 32 float 단위)
//   - MISALIGN_FLOATS = 1  → 4B misaligned (tile 시작 주소가 128B+4B, 두 segment 걸침)
// =======================

template<int MISALIGN_FLOATS>
__global__ void cp_async_align_kernel(const float* __restrict__ g_in,
                                      float* __restrict__ g_out,
                                      int tiles_per_warp)
{
    __shared__ float sh[WARP_SIZE];  // 32 floats = 128B

    int lane    = threadIdx.x;       // 0..31
    int warp_id = blockIdx.x;        // block당 1 warp

    // 이 warp가 담당하는 global base offset
    int warp_base = warp_id * tiles_per_warp * WARP_SIZE;

    float acc = 0.0f;

    // 각 warp가 tiles_per_warp 개의 tile을 순서대로 처리
    for (int t = 0; t < tiles_per_warp; ++t) {
        int tile_offset  = warp_base + t * WARP_SIZE;
        int global_index = tile_offset + lane + MISALIGN_FLOATS;

        const float* src = g_in + global_index;
        float* dst       = &sh[lane];

        cp_async_4B(dst, src);
        cp_async_commit();
        cp_async_wait_all();   // 비동기성은 안 쓰고, transaction 패턴만 비교

        float v = sh[lane];
        acc = acc * 1.0001f + v;  // 약간의 연산 (dead-code 방지)
    }

    // warp별 결과 저장
    int out_idx = warp_id * WARP_SIZE + lane;
    g_out[out_idx] = acc;
}

// =======================
// host util
// =======================

void init_data(float* a, int n)
{
    for (int i = 0; i < n; ++i) {
        a[i] = (i % 31) * 0.1f;
    }
}

// =======================
// main
// =======================

int main()
{
    printf("=== cp.async Test 3: alignment (128B aligned vs misaligned) ===\n");
    printf("THREADS_PER_BLOCK = %d, BLOCKS = %d\n", THREADS_PER_BLOCK, BLOCKS);
    printf("TILES_PER_WARP    = %d\n", TILES_PER_WARP);
    printf("TOTAL_FLOATS      = %d\n\n", TOTAL_FLOATS);

    size_t bytes_in  = TOTAL_FLOATS * sizeof(float);
    size_t bytes_out = BLOCKS * WARP_SIZE * sizeof(float);

    float* h_in  = (float*)malloc(bytes_in);
    float* h_out = (float*)malloc(bytes_out);

    init_data(h_in, TOTAL_FLOATS);

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, bytes_in));
    CHECK_CUDA(cudaMalloc(&d_out, bytes_out));

    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes_in, cudaMemcpyHostToDevice));

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCKS);

    // -------------------------
    // 1) 128B 정렬된 cp.async (MISALIGN_FLOATS = 0)
    // -------------------------
    {
        printf("[aligned] warm-up + timing\n");
        cp_async_align_kernel<0><<<grid, block>>>(d_in, d_out, TILES_PER_WARP);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        cp_async_align_kernel<0><<<grid, block>>>(d_in, d_out, TILES_PER_WARP);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes_out, cudaMemcpyDeviceToHost));

        printf("[aligned] kernel time: %.3f ms\n\n", ms);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    // -------------------------
    // 2) 4B misaligned cp.async (MISALIGN_FLOATS = 1)
    // -------------------------
    {
        printf("[misaligned] warm-up + timing\n");
        cp_async_align_kernel<1><<<grid, block>>>(d_in, d_out, TILES_PER_WARP);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        cp_async_align_kernel<1><<<grid, block>>>(d_in, d_out, TILES_PER_WARP);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes_out, cudaMemcpyDeviceToHost));

        printf("[misaligned] kernel time: %.3f ms\n\n", ms);

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
nvcc -O3 -arch=sm_86 -lineinfo -o cp_async_alignment_test.exe cp_async_alignment_test.cu

ncu --kernel-name regex:cp_async_align_kernel.* --metrics dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum   ./cp_async_alignment_test.exe


*/