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
constexpr int THREADS_PER_BLOCK = 32;     // 1 warp / block
constexpr int BLOCKS            = 80;
constexpr int ITERS             = 100000;

// 총 thread 수
constexpr int TOTAL_THREADS     = THREADS_PER_BLOCK * BLOCKS;
constexpr int TOTAL_ELEMS       = TOTAL_THREADS;

// =======================
// 호스트 유틸
// =======================

void init_data(float* a, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] = (i % 13) * 0.1f;
    }
}

// =======================
// 1. Normal shared load kernel
//    shared[tid] = global[idx]
//    __syncthreads() 로 membar 사용
// =======================
__global__ void normal_shared_load_kernel(const float* __restrict__ g_in,
                                          float* __restrict__ g_out,
                                          int total_elems)
{
    __shared__ float sh[WARP_SIZE];  // 32

    int tid     = threadIdx.x;                       // 0..31
    int gid     = blockIdx.x * blockDim.x + tid;     // global index

    if (gid >= total_elems) return;

    float acc = 0.0f;

    // 같은 위치를 여러 번 반복해서 읽어서 stall을 키움
    for (int it = 0; it < ITERS; ++it) {
        float v = g_in[gid];

        // shared에 쓴 후 전체 block barrier
        sh[tid] = v;
        __syncthreads();

        // 이웃 lane 값을 읽어서 "진짜 공유"를 강제
        int neighbor = (tid + 1) & (WARP_SIZE - 1);
        float w = sh[neighbor];
        acc += w;

        __syncthreads();
    }

    g_out[gid] = acc;
}

// =======================
// 2. cp.async 기반 shared load kernel
//    cp.async.shared.global + commit_group + wait_group
//    단일 warp만 사용하므로 __syncthreads()는 없음
// =======================

__device__ __forceinline__ void cp_async_4B(void* smem_ptr, const void* gmem_ptr) {
#if __CUDA_ARCH__ >= 800
    // shared 주소를 32-bit 공간으로 변환
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(smem_addr), "l"(gmem_ptr)
    );
#else
    // fallback (pre-SM80): 그냥 ld/st
    *reinterpret_cast<float*>(smem_ptr) = *reinterpret_cast<const float*>(gmem_ptr);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if __CUDA_ARCH__ >= 800
    // 모든 outstanding group이 완료될 때까지 대기
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

__global__ void cp_async_shared_load_kernel(const float* __restrict__ g_in,
                                            float* __restrict__ g_out,
                                            int total_elems)
{
    __shared__ float sh[WARP_SIZE];   // 32

    int tid = threadIdx.x;                           // 0..31
    int gid = blockIdx.x * blockDim.x + tid;         // global index

    if (gid >= total_elems) return;

    float acc = 0.0f;

    // 단일 warp만 사용하므로 warp-scope cp.async만으로 동기화 가능
    for (int it = 0; it < ITERS; ++it) {
        const float* src = g_in + gid;
        float* dst       = sh + tid;

        // 각 lane이 자기 위치로 4B cp.async
        cp_async_4B(dst, src);
        cp_async_commit();
        cp_async_wait_all();  // 이 시점 이후 sh[tid] 사용 가능

        // block barrier 없이도 warp 내에서는 안전
        int neighbor = (tid + 1) & (WARP_SIZE - 1);
        float w = sh[neighbor];
        acc += w;
    }

    g_out[gid] = acc;
}

// =======================
// main
// =======================

int main()
{
    printf("=== cp.async Test 1: normal load vs cp.async load ===\n");
    printf("THREADS_PER_BLOCK = %d, BLOCKS = %d, ITERS = %d\n",
           THREADS_PER_BLOCK, BLOCKS, ITERS);
    printf("Total threads / elems = %d\n\n", TOTAL_THREADS);

    size_t bytes = TOTAL_ELEMS * sizeof(float);

    float* h_in  = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);

    init_data(h_in, TOTAL_ELEMS);

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));

    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCKS);

    // -------------------------
    // 1) normal shared load
    // -------------------------
    {
        printf("[normal] warm-up + timing\n");
        normal_shared_load_kernel<<<grid, block>>>(d_in, d_out, TOTAL_ELEMS);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        normal_shared_load_kernel<<<grid, block>>>(d_in, d_out, TOTAL_ELEMS);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

        printf("[normal] kernel time: %.3f ms\n\n", ms);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    // -------------------------
    // 2) cp.async shared load
    // -------------------------
    {
        printf("[cp.async] warm-up + timing\n");
        cp_async_shared_load_kernel<<<grid, block>>>(d_in, d_out, TOTAL_ELEMS);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        cp_async_shared_load_kernel<<<grid, block>>>(d_in, d_out, TOTAL_ELEMS);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

        printf("[cp.async] kernel time: %.3f ms\n\n", ms);

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
nvcc -O3 -arch=sm_86 -lineinfo -o cp_async_load_test.exe cp_async_load_test.cu



# normal load 버전
ncu --kernel-name regex:normal_shared_load_kernel.* --metrics smsp__warp_issue_stalled_membar_per_warp_active.avg,smsp__warp_issue_stalled_membar_per_warp_active.pct,smsp__warp_issue_stalled_memory_dependency_per_warp_active.avg,smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct   ./cp_async_load_test.exe

# cp.async 버전
ncu --kernel-name regex:cp_async_shared_load_kernel.* --metrics smsp__warp_issue_stalled_membar_per_warp_active.avg,smsp__warp_issue_stalled_membar_per_warp_active.pct,smsp__warp_issue_stalled_memory_dependency_per_warp_active.avg,smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct   ./cp_async_load_test.exe

*/