// occupancy_reg_pressure_test_v3.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                   \
                    cudaGetErrorString(err__), __FILE__, __LINE__);        \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

// ---------------------------------------------------------------------
// 공통: 약간 무거운 연산
// ---------------------------------------------------------------------
__device__ float do_work(float x, int iters)
{
    for (int i = 0; i < iters; ++i) {
        x = x * 1.000001f + 1.0f;
        x = x - 1.0f;
    }
    return x;
}

// ---------------------------------------------------------------------
// 레지스터 적게 쓰는 버전
// ---------------------------------------------------------------------
__global__ void kernel_low_regs(float* __restrict__ out,
                                const float* __restrict__ in,
                                int iters)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float x = in[idx];

    float acc = x;
    acc = do_work(acc, iters);

    out[idx] = acc;
}

// ---------------------------------------------------------------------
// 레지스터 엄청 많이 쓰는 버전
//  - scalar 96개 정도 선언해서 numRegs를 확 늘림
//  - block_size=256일 때 레지스터가 maxActiveBlocks/SM을 제한하도록 유도
// ---------------------------------------------------------------------
__global__ void kernel_high_regs(float* __restrict__ out,
                                 const float* __restrict__ in,
                                 int iters)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float x = in[idx];

    // 레지스터를 폭발적으로 사용
    float a0  = x + 0.0f,   a1  = x + 1.0f,   a2  = x + 2.0f,   a3  = x + 3.0f;
    float a4  = x + 4.0f,   a5  = x + 5.0f,   a6  = x + 6.0f,   a7  = x + 7.0f;
    float a8  = x + 8.0f,   a9  = x + 9.0f,   a10 = x + 10.0f,  a11 = x + 11.0f;
    float a12 = x + 12.0f,  a13 = x + 13.0f,  a14 = x + 14.0f,  a15 = x + 15.0f;
    float a16 = x + 16.0f,  a17 = x + 17.0f,  a18 = x + 18.0f,  a19 = x + 19.0f;
    float a20 = x + 20.0f,  a21 = x + 21.0f,  a22 = x + 22.0f,  a23 = x + 23.0f;
    float a24 = x + 24.0f,  a25 = x + 25.0f,  a26 = x + 26.0f,  a27 = x + 27.0f;
    float a28 = x + 28.0f,  a29 = x + 29.0f,  a30 = x + 30.0f,  a31 = x + 31.0f;

    float b0  = x + 32.0f,  b1  = x + 33.0f,  b2  = x + 34.0f,  b3  = x + 35.0f;
    float b4  = x + 36.0f,  b5  = x + 37.0f,  b6  = x + 38.0f,  b7  = x + 39.0f;
    float b8  = x + 40.0f,  b9  = x + 41.0f,  b10 = x + 42.0f,  b11 = x + 43.0f;
    float b12 = x + 44.0f,  b13 = x + 45.0f,  b14 = x + 46.0f,  b15 = x + 47.0f;
    float b16 = x + 48.0f,  b17 = x + 49.0f,  b18 = x + 50.0f,  b19 = x + 51.0f;
    float b20 = x + 52.0f,  b21 = x + 53.0f,  b22 = x + 54.0f,  b23 = x + 55.0f;
    float b24 = x + 56.0f,  b25 = x + 57.0f,  b26 = x + 58.0f,  b27 = x + 59.0f;
    float b28 = x + 60.0f,  b29 = x + 61.0f,  b30 = x + 62.0f,  b31 = x + 63.0f;

    float c0  = x + 64.0f,  c1  = x + 65.0f,  c2  = x + 66.0f,  c3  = x + 67.0f;
    float c4  = x + 68.0f,  c5  = x + 69.0f,  c6  = x + 70.0f,  c7  = x + 71.0f;
    float c8  = x + 72.0f,  c9  = x + 73.0f,  c10 = x + 74.0f,  c11 = x + 75.0f;
    float c12 = x + 76.0f,  c13 = x + 77.0f,  c14 = x + 78.0f,  c15 = x + 79.0f;
    float c16 = x + 80.0f,  c17 = x + 81.0f,  c18 = x + 82.0f,  c19 = x + 83.0f;
    float c20 = x + 84.0f,  c21 = x + 85.0f,  c22 = x + 86.0f,  c23 = x + 87.0f;
    float c24 = x + 88.0f,  c25 = x + 89.0f,  c26 = x + 90.0f,  c27 = x + 91.0f;
    float c28 = x + 92.0f,  c29 = x + 93.0f,  c30 = x + 94.0f,  c31 = x + 95.0f;

    for (int i = 0; i < iters; ++i) {
        a0  = do_work(a0,  1); a1  = do_work(a1,  1); a2  = do_work(a2,  1); a3  = do_work(a3,  1);
        a4  = do_work(a4,  1); a5  = do_work(a5,  1); a6  = do_work(a6,  1); a7  = do_work(a7,  1);
        a8  = do_work(a8,  1); a9  = do_work(a9,  1); a10 = do_work(a10, 1); a11 = do_work(a11, 1);
        a12 = do_work(a12, 1); a13 = do_work(a13, 1); a14 = do_work(a14, 1); a15 = do_work(a15, 1);
        a16 = do_work(a16, 1); a17 = do_work(a17, 1); a18 = do_work(a18, 1); a19 = do_work(a19, 1);
        a20 = do_work(a20, 1); a21 = do_work(a21, 1); a22 = do_work(a22, 1); a23 = do_work(a23, 1);
        a24 = do_work(a24, 1); a25 = do_work(a25, 1); a26 = do_work(a26, 1); a27 = do_work(a27, 1);
        a28 = do_work(a28, 1); a29 = do_work(a29, 1); a30 = do_work(a30, 1); a31 = do_work(a31, 1);

        b0  = do_work(b0,  1); b1  = do_work(b1,  1); b2  = do_work(b2,  1); b3  = do_work(b3,  1);
        b4  = do_work(b4,  1); b5  = do_work(b5,  1); b6  = do_work(b6,  1); b7  = do_work(b7,  1);
        b8  = do_work(b8,  1); b9  = do_work(b9,  1); b10 = do_work(b10, 1); b11 = do_work(b11, 1);
        b12 = do_work(b12, 1); b13 = do_work(b13, 1); b14 = do_work(b14, 1); b15 = do_work(b15, 1);
        b16 = do_work(b16, 1); b17 = do_work(b17, 1); b18 = do_work(b18, 1); b19 = do_work(b19, 1);
        b20 = do_work(b20, 1); b21 = do_work(b21, 1); b22 = do_work(b22, 1); b23 = do_work(b23, 1);
        b24 = do_work(b24, 1); b25 = do_work(b25, 1); b26 = do_work(b26, 1); b27 = do_work(b27, 1);
        b28 = do_work(b28, 1); b29 = do_work(b29, 1); b30 = do_work(b30, 1); b31 = do_work(b31, 1);

        c0  = do_work(c0,  1); c1  = do_work(c1,  1); c2  = do_work(c2,  1); c3  = do_work(c3,  1);
        c4  = do_work(c4,  1); c5  = do_work(c5,  1); c6  = do_work(c6,  1); c7  = do_work(c7,  1);
        c8  = do_work(c8,  1); c9  = do_work(c9,  1); c10 = do_work(c10, 1); c11 = do_work(c11, 1);
        c12 = do_work(c12, 1); c13 = do_work(c13, 1); c14 = do_work(c14, 1); c15 = do_work(c15, 1);
        c16 = do_work(c16, 1); c17 = do_work(c17, 1); c18 = do_work(c18, 1); c19 = do_work(c19, 1);
        c20 = do_work(c20, 1); c21 = do_work(c21, 1); c22 = do_work(c22, 1); c23 = do_work(c23, 1);
        c24 = do_work(c24, 1); c25 = do_work(c25, 1); c26 = do_work(c26, 1); c27 = do_work(c27, 1);
        c28 = do_work(c28, 1); c29 = do_work(c29, 1); c30 = do_work(c30, 1); c31 = do_work(c31, 1);
    }

    float acc = 0.0f;
    acc += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    acc += a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    acc += a16 + a17 + a18 + a19 + a20 + a21 + a22 + a23;
    acc += a24 + a25 + a26 + a27 + a28 + a29 + a30 + a31;

    acc += b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7;
    acc += b8 + b9 + b10 + b11 + b12 + b13 + b14 + b15;
    acc += b16 + b17 + b18 + b19 + b20 + b21 + b22 + b23;
    acc += b24 + b25 + b26 + b27 + b28 + b29 + b30 + b31;

    acc += c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
    acc += c8 + c9 + c10 + c11 + c12 + c13 + c14 + c15;
    acc += c16 + c17 + c18 + c19 + c20 + c21 + c22 + c23;
    acc += c24 + c25 + c26 + c27 + c28 + c29 + c30 + c31;

    out[idx] = acc;
}

// ---------------------------------------------------------------------
// Occupancy 정보 출력
// ---------------------------------------------------------------------
template <typename KernelFunc>
void print_kernel_info(const char* name, KernelFunc func,
                       int block_size, size_t dyn_shared_bytes = 0)
{
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, (const void*)func));

    printf("=== %s ===\n", name);
    printf("  numRegs          : %d\n", attr.numRegs);
    printf("  sharedSizeBytes  : %zu (static)\n", (size_t)attr.sharedSizeBytes);
    printf("  localSizeBytes   : %zu\n", (size_t)attr.localSizeBytes);
    printf("  maxThreadsPerBlk : %d\n", attr.maxThreadsPerBlock);

    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int maxActiveBlocks;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, func, block_size, dyn_shared_bytes));

    int warpsPerBlock = (block_size + prop.warpSize - 1) / prop.warpSize;
    int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    int activeWarps   = maxActiveBlocks * warpsPerBlock;

    float occupancy = (float)activeWarps / (float)maxWarpsPerSM;

    printf("  SM max threads     : %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  block_size         : %d\n", block_size);
    printf("  maxActiveBlocks/SM : %d\n", maxActiveBlocks);
    printf("  activeWarps/SM     : %d (max %d)\n", activeWarps, maxWarpsPerSM);
    printf("  occupancy (approx) : %.2f\n\n", occupancy);
}

// ---------------------------------------------------------------------
// 타이밍 유틸
// ---------------------------------------------------------------------
float run_and_time(void (*kernel)(float*, const float*, int),
                   const char* name,
                   float* d_out, const float* d_in,
                   int num_elems, int iters,
                   dim3 grid, dim3 block,
                   int repeat = 10)
{
    for (int i = 0; i < 3; ++i) {
        kernel<<<grid, block>>>(d_out, d_in, iters);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i) {
        kernel<<<grid, block>>>(d_out, d_in, iters);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    ms /= repeat;

    printf("[%s] avg time: %.4f ms (iters=%d, elems=%d)\n",
           name, ms, iters, num_elems);

    return ms;
}

// ---------------------------------------------------------------------
// main
// ---------------------------------------------------------------------
int main()
{
    const int block_size = 256;
    const int num_blocks = 4096;              // 총 1,048,576 threads
    const int num_elems  = block_size * num_blocks;

    const int iters_low  = 2000;
    const int iters_high = 32;                // 레지스터 96개 × 32 ≈ 3000 “유효 반복”

    printf("num_elems = %d\n", num_elems);

    float* h_in  = (float*)malloc(num_elems * sizeof(float));
    float* h_out = (float*)malloc(num_elems * sizeof(float));
    for (int i = 0; i < num_elems; ++i) {
        h_in[i] = (float)i * 0.001f;
    }

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  num_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, num_elems * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in,
                          num_elems * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 block(block_size);
    dim3 grid(num_blocks);

    print_kernel_info("kernel_low_regs",  kernel_low_regs,  block_size);
    print_kernel_info("kernel_high_regs", kernel_high_regs, block_size);

    run_and_time(kernel_low_regs,  "low_regs",
                 d_out, d_in, num_elems, iters_low,  grid, block);
    run_and_time(kernel_high_regs, "high_regs",
                 d_out, d_in, num_elems, iters_high, grid, block);

    CUDA_CHECK(cudaMemcpy(h_out, d_out,
                          num_elems * sizeof(float),
                          cudaMemcpyDeviceToHost));

    printf("sample output: h_out[0] = %f\n", h_out[0]);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
