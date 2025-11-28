// occupancy_reg_pressure_test.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                   \
                    cudaGetErrorString(err__), __FILE__, __LINE__);        \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

// ------------------------------------------------------------
// 공통: 약간 무거운 연산
// ------------------------------------------------------------
__device__ float do_work(float x, int iters)
{
    // 가벼운 FMA 반복
    for (int i = 0; i < iters; ++i) {
        x = x * 1.000001f + 1.0f;
        x = x - 1.0f;
    }
    return x;
}

// ------------------------------------------------------------
// (1) 레지스터 사용 적은 커널
//  - 스칼라 몇 개만 사용
//  - occupancy 상대적으로 높게 나올 가능성
// ------------------------------------------------------------
__global__ void kernel_low_regs(float* __restrict__ out,
                                const float* __restrict__ in,
                                int iters)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float x = in[idx];

    // 레지스터 몇 개만 쓰는 단순 버전
    float acc = x;
    acc = do_work(acc, iters);

    out[idx] = acc;
}

// ------------------------------------------------------------
// (2) 레지스터 사용 높은 커널
//
//  - 의도적으로 많은 스칼라 레지스터를 만들어서
//    register pressure를 올림
//  - 컴파일러가 최적화해버리지 않도록 모든 변수에
//    반복적으로 연산을 수행
// ------------------------------------------------------------
__global__ void kernel_high_regs(float* __restrict__ out,
                                 const float* __restrict__ in,
                                 int iters)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float x = in[idx];

    // 스칼라 레지스터를 많이 사용
    // (컴파일러가 어느 정도 최적화하겠지만,
    //  일반적으로 low_regs 보다 훨씬 많은 레지스터 사용)
    float a0  = x,  a1  = x + 1.0f,  a2  = x + 2.0f,  a3  = x + 3.0f;
    float a4  = x + 4.0f,  a5  = x + 5.0f,  a6  = x + 6.0f,  a7  = x + 7.0f;
    float a8  = x + 8.0f,  a9  = x + 9.0f,  a10 = x + 10.0f, a11 = x + 11.0f;
    float a12 = x + 12.0f, a13 = x + 13.0f, a14 = x + 14.0f, a15 = x + 15.0f;
    float a16 = x + 16.0f, a17 = x + 17.0f, a18 = x + 18.0f, a19 = x + 19.0f;
    float a20 = x + 20.0f, a21 = x + 21.0f, a22 = x + 22.0f, a23 = x + 23.0f;
    float a24 = x + 24.0f, a25 = x + 25.0f, a26 = x + 26.0f, a27 = x + 27.0f;
    float a28 = x + 28.0f, a29 = x + 29.0f, a30 = x + 30.0f, a31 = x + 31.0f;

    for (int i = 0; i < iters; ++i) {
        // 각 변수에 일을 시켜서 레지스터가 살아있게 유지
        a0  = do_work(a0,  1);
        a1  = do_work(a1,  1);
        a2  = do_work(a2,  1);
        a3  = do_work(a3,  1);
        a4  = do_work(a4,  1);
        a5  = do_work(a5,  1);
        a6  = do_work(a6,  1);
        a7  = do_work(a7,  1);
        a8  = do_work(a8,  1);
        a9  = do_work(a9,  1);
        a10 = do_work(a10, 1);
        a11 = do_work(a11, 1);
        a12 = do_work(a12, 1);
        a13 = do_work(a13, 1);
        a14 = do_work(a14, 1);
        a15 = do_work(a15, 1);
        a16 = do_work(a16, 1);
        a17 = do_work(a17, 1);
        a18 = do_work(a18, 1);
        a19 = do_work(a19, 1);
        a20 = do_work(a20, 1);
        a21 = do_work(a21, 1);
        a22 = do_work(a22, 1);
        a23 = do_work(a23, 1);
        a24 = do_work(a24, 1);
        a25 = do_work(a25, 1);
        a26 = do_work(a26, 1);
        a27 = do_work(a27, 1);
        a28 = do_work(a28, 1);
        a29 = do_work(a29, 1);
        a30 = do_work(a30, 1);
        a31 = do_work(a31, 1);
    }

    // 결과를 하나로 합쳐서 out에 저장 (dead-code 제거 방지)
    float acc =
        a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 +
        a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 +
        a16 + a17 + a18 + a19 + a20 + a21 + a22 + a23 +
        a24 + a25 + a26 + a27 + a28 + a29 + a30 + a31;

    out[idx] = acc;
}

// ------------------------------------------------------------
// Occupancy 정보 출력 유틸
// ------------------------------------------------------------
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
    printf("  occupancy (approx) : %.2f\n", occupancy);
    printf("\n");
}

// ------------------------------------------------------------
// 타이밍 유틸
// ------------------------------------------------------------
float run_and_time(void (*kernel)(float*, const float*, int),
                   const char* name,
                   float* d_out, const float* d_in,
                   int num_elems, int iters,
                   dim3 grid, dim3 block,
                   int repeat = 10)
{
    // warmup
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

    ms /= repeat; // 평균

    printf("[%s] avg time: %.4f ms (iters=%d, elems=%d)\n",
           name, ms, iters, num_elems);

    return ms;
}

// ------------------------------------------------------------
// main
// ------------------------------------------------------------
int main()
{
    // 실험 파라미터
    const int block_size = 256;
    const int num_blocks = 4096;    // 총 thread = 4096 * 256 ≈ 1M
    const int num_elems  = block_size * num_blocks;
    const int iters_low  = 2000;    // low_regs용 연산량
    const int iters_high = 2000;    // high_regs용 (필요시 조절)

    printf("num_elems = %d\n", num_elems);

    // 입력/출력 버퍼
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

    // 커널 속성 / occupancy 정보 출력
    print_kernel_info("kernel_low_regs",  kernel_low_regs,  block_size);
    print_kernel_info("kernel_high_regs", kernel_high_regs, block_size);

    // 타이밍 실행
    run_and_time(kernel_low_regs,  "low_regs",
                 d_out, d_in, num_elems, iters_low,  grid, block);
    run_and_time(kernel_high_regs, "high_regs",
                 d_out, d_in, num_elems, iters_high, grid, block);

    // 결과 하나 정도 읽어서 dead-code 최적화 방지 확인
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

// nvcc -O3 -arch=sm_86 occupancy_reg_pressure_test.cu -o occ_test
