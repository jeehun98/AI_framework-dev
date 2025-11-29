#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr,"CUDA Error %s:%d : %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)
#endif

// =====================================================
// Warp scan using shfl_up
// =====================================================
__inline__ __device__
unsigned warpScanInclusive(unsigned val)
{
    unsigned mask = 0xffffffff;
    for (int offset = 1; offset < 32; offset <<= 1) {
        unsigned y = __shfl_up_sync(mask, val, offset);
        if (threadIdx.x % 32 >= offset) val += y;
    }
    return val;
}

// =====================================================
// Kernel 1: Shared-memory Blelloch scan (exclusive scan)
// =====================================================
__global__
void scan_shared(const unsigned* __restrict__ g_in,
                 unsigned* __restrict__ g_out,
                 int N)
{
    extern __shared__ unsigned smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    unsigned val = (gid < N ? g_in[gid] : 0);
    smem[tid] = val;
    __syncthreads();

    // ------- up-sweep -------
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < blockDim.x)
            smem[idx] += smem[idx - offset];
        __syncthreads();
    }

    // clear last element for exclusive scan
    if (tid == blockDim.x - 1)
        smem[tid] = 0;
    __syncthreads();

    // ------- down-sweep -------
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < blockDim.x) {
            unsigned t = smem[idx - offset];
            smem[idx - offset] = smem[idx];
            smem[idx] += t;
        }
        __syncthreads();
    }

    if (gid < N)
        g_out[gid] = smem[tid];
}

// =====================================================
// Kernel 2: Warp-level shuffle scan
// =====================================================
__global__
void scan_warp(const unsigned* __restrict__ g_in,
               unsigned* __restrict__ g_out,
               int N)
{
    extern __shared__ unsigned warpSums[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    unsigned val = (gid < N ? g_in[gid] : 0);

    int lane   = tid & 31;
    int warpId = tid >> 5;

    // (1) warp-level inclusive scan
    unsigned scan = warpScanInclusive(val);

    // (2) warp ending values -> shared
    if (lane == 31)
        warpSums[warpId] = scan;
    __syncthreads();

    // (3) warp 0 scans warpSums[]
    if (warpId == 0) {
        unsigned s = warpScanInclusive(warpSums[lane]);
        warpSums[lane] = s;
    }
    __syncthreads();

    // (4) distribute prefix corrections
    if (warpId > 0)
        scan += warpSums[warpId - 1];

    if (gid < N)
        g_out[gid] = scan - val;   // convert incl->exclusive
}

// =====================================================
// CPU reference
// =====================================================
void cpu_scan(const unsigned* in, unsigned* out, int N)
{
    unsigned s = 0;
    for (int i = 0; i < N; i++) {
        out[i] = s;
        s += in[i];
    }
}

// =====================================================
// Timing wrapper
// =====================================================
float run_kernel(void(*kernel)(const unsigned*, unsigned*, int),
                 const unsigned* d_in, unsigned* d_out,
                 int N, int blocks, int threads,
                 size_t smem, int iters)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < 5; i++)
        kernel<<<blocks, threads, smem>>>(d_in, d_out, N);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        kernel<<<blocks, threads, smem>>>(d_in, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms / iters;
}

// =====================================================
// main
// =====================================================
int main(int argc, char** argv)
{
    int N = 1 << 20;
    int threads = 1024;
    int blocks  = (N + threads - 1) / threads;
    int iters = 100;

    printf("N=%d, blocks=%d, threads=%d\n", N, blocks, threads);

    // host input
    unsigned* h_in  = (unsigned*)malloc(N * sizeof(unsigned));
    unsigned* h_out = (unsigned*)malloc(N * sizeof(unsigned));
    for (int i = 0; i < N; i++) h_in[i] = 1;

    // device alloc
    unsigned *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(unsigned), cudaMemcpyHostToDevice));

    // shared memory
    size_t smem_shared = threads * sizeof(unsigned);     // for Blelloch
    size_t smem_warp   = (threads / 32) * sizeof(unsigned);

    float t_shared = run_kernel(scan_shared, d_in, d_out,
                                N, blocks, threads,
                                smem_shared, iters);

    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(unsigned), cudaMemcpyDeviceToHost));
    unsigned* ref = (unsigned*)malloc(N * sizeof(unsigned));
    cpu_scan(h_in, ref, N);

    // check
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (h_out[i] != ref[i]) { ok = false; break; }
    printf("[shared] correct = %d\n", ok);

    float t_warp = run_kernel(scan_warp, d_in, d_out,
                              N, blocks, threads,
                              smem_warp, iters);

    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(unsigned), cudaMemcpyDeviceToHost));
    ok = true;
    for (int i = 0; i < N; i++)
        if (h_out[i] != ref[i]) { ok = false; break; }
    printf("[warp]   correct = %d\n", ok);

    printf("\n=== Timing ===\n");
    printf("shared scan : %.4f ms\n", t_shared);
    printf("warp scan   : %.4f ms\n", t_warp);
    printf("Speedup     : %.2fx\n", t_shared / t_warp);

    free(h_in);
    free(h_out);
    free(ref);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

/*
nvcc -O3 -arch=sm_86 scan_test.cu -o scan_test

ncu --kernel-name regex:scan_shared.*     --launch-skip 5 --launch-count 1 --set full     ./scan_test

ncu --kernel-name regex:scan_warp.*     --launch-skip 5 --launch-count 1 --set full     ./scan_test

*/
