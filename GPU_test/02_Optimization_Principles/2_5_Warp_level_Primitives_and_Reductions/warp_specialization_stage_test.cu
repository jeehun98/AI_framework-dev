#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err__));          \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

constexpr int WARP_SIZE   = 32;
constexpr int TILE_SIZE   = 128;   // per block tile size (must divide N)
constexpr int THREADS_2ST = 64;    // 2 warps: load+store, compute
constexpr int THREADS_3ST = 96;    // 3 warps: load, compute, store

// 간단한 연산: out = a * b + 1.0f
__device__ __forceinline__ float op(float a, float b) {
    return a * b + 1.0f;
}

// 2-stage warp specialization
// warp0: load + store
// warp1: compute
__global__ void ws2_kernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ out,
                           int N)
{
    extern __shared__ float smem[];
    float* sm_in  = smem;                 // [TILE_SIZE]
    float* sm_out = smem + TILE_SIZE;     // [TILE_SIZE]

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane    = threadIdx.x % WARP_SIZE;

    int tile_idx   = blockIdx.x;
    int base_index = tile_idx * TILE_SIZE;

    if (base_index >= N) return;

    // --- Stage 1: load (warp0) ---
    if (warp_id == 0) {
        for (int i = lane; i < TILE_SIZE; i += WARP_SIZE) {
            int idx = base_index + i;
            float va = (idx < N) ? a[idx] : 0.0f;
            float vb = (idx < N) ? b[idx] : 0.0f;
            sm_in[i] = op(va, vb);  // 바로 op 적용해서 다음 stage에 넘겨도 됨
        }
    }
    __syncthreads();

    // --- Stage 2: compute (warp1) ---
    if (warp_id == 1) {
        for (int i = lane; i < TILE_SIZE; i += WARP_SIZE) {
            float v = sm_in[i];
            // dummy extra compute
            v = v * 1.0001f + 0.1f;
            sm_out[i] = v;
        }
    }
    __syncthreads();

    // --- Stage 3: store (warp0) ---
    if (warp_id == 0) {
        for (int i = lane; i < TILE_SIZE; i += WARP_SIZE) {
            int idx = base_index + i;
            if (idx < N) {
                out[idx] = sm_out[i];
            }
        }
    }
}

// 3-stage warp specialization
// warp0: load
// warp1: compute
// warp2: store
__global__ void ws3_kernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ out,
                           int N)
{
    extern __shared__ float smem[];
    float* sm_in  = smem;                 // [TILE_SIZE]
    float* sm_out = smem + TILE_SIZE;     // [TILE_SIZE]

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane    = threadIdx.x % WARP_SIZE;

    int tile_idx   = blockIdx.x;
    int base_index = tile_idx * TILE_SIZE;

    if (base_index >= N) return;

    // --- Stage 1: load (warp0) ---
    if (warp_id == 0) {
        for (int i = lane; i < TILE_SIZE; i += WARP_SIZE) {
            int idx = base_index + i;
            float va = (idx < N) ? a[idx] : 0.0f;
            float vb = (idx < N) ? b[idx] : 0.0f;
            sm_in[i] = op(va, vb);
        }
    }
    __syncthreads();

    // --- Stage 2: compute (warp1) ---
    if (warp_id == 1) {
        for (int i = lane; i < TILE_SIZE; i += WARP_SIZE) {
            float v = sm_in[i];
            // dummy extra compute
            v = v * 1.0001f + 0.1f;
            sm_out[i] = v;
        }
    }
    __syncthreads();

    // --- Stage 3: store (warp2) ---
    if (warp_id == 2) {
        for (int i = lane; i < TILE_SIZE; i += WARP_SIZE) {
            int idx = base_index + i;
            if (idx < N) {
                out[idx] = sm_out[i];
            }
        }
    }
}

float run_kernel_2stage(const float* d_a,
                        const float* d_b,
                        float* d_out,
                        int N,
                        int iters)
{
    int grid = N / TILE_SIZE;
    size_t shmem_bytes = sizeof(float) * TILE_SIZE * 2;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // warmup
    ws2_kernel<<<grid, THREADS_2ST, shmem_bytes>>>(d_a, d_b, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        ws2_kernel<<<grid, THREADS_2ST, shmem_bytes>>>(d_a, d_b, d_out, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / iters;
}

float run_kernel_3stage(const float* d_a,
                        const float* d_b,
                        float* d_out,
                        int N,
                        int iters)
{
    int grid = N / TILE_SIZE;
    size_t shmem_bytes = sizeof(float) * TILE_SIZE * 2;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // warmup
    ws3_kernel<<<grid, THREADS_3ST, shmem_bytes>>>(d_a, d_b, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        ws3_kernel<<<grid, THREADS_3ST, shmem_bytes>>>(d_a, d_b, d_out, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / iters;
}

int main(int argc, char** argv)
{
    int N     = 1 << 20;   // 1,048,576
    int iters = 100;
    if (argc > 1) {
        iters = std::atoi(argv[1]);
    }

    printf("N=%d, tile=%d, grid=%d\n", N, TILE_SIZE, N / TILE_SIZE);

    size_t bytes = sizeof(float) * N;
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);

    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_out;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    float t2 = run_kernel_2stage(d_a, d_b, d_out, N, iters);
    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    double checksum2 = 0.0;
    for (int i = 0; i < 1024; ++i) checksum2 += h_out[i];

    float t3 = run_kernel_3stage(d_a, d_b, d_out, N, iters);
    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    double checksum3 = 0.0;
    for (int i = 0; i < 1024; ++i) checksum3 += h_out[i];

    printf("checksum 2-stage = %.6f\n", checksum2);
    printf("checksum 3-stage = %.6f\n", checksum3);

    printf("\n=== Timing (avg over %d iters) ===\n", iters);
    printf("2-stage (warp0: load+store, warp1: compute): %.4f ms\n", t2);
    printf("3-stage (warp0: load, warp1: compute, warp2: store): %.4f ms\n", t3);
    printf("Speedup (2-stage / 3-stage): %.2fx\n", t2 / t3);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_out));
    free(h_a);
    free(h_b);
    free(h_out);

    return 0;
}
/*
nvcc -O3 -std=c++17   -arch=sm_86   warp_specialization_stage_test.cu   -o warp_specialization_stage_test.exe

# 2-stage
ncu --kernel-name regex:ws2_kernel.*     --launch-skip 5 --launch-count 1     --set full     ./warp_specialization_stage_test.exe 100

# 3-stage
ncu --kernel-name regex:ws3_kernel.*     --launch-skip 5 --launch-count 1     --set full     ./warp_specialization_stage_test.exe 100

*/