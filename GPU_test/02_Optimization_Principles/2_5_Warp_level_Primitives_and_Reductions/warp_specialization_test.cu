#include <cstdio>
#include <cuda_runtime.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA Error %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(1);                                                   \
        }                                                              \
    } while (0)
#endif

// -------------------------------------------
// Monolithic (load + compute in same warp)
// -------------------------------------------
__global__ void mono_kernel(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid * 4 >= N) return;

    float sum = 0.f;
    #pragma unroll 4
    for (int i = 0; i < 4; i++)
    {
        float a = A[tid * 4 + i];   // global load
        float b = B[tid * 4 + i];
        sum += a * b;               // compute
    }
    C[tid] = sum;
}

// -------------------------------------------
// Warp-specialization version (warp0=loader, warp1=compute)
// Using cp.async to reduce stall
// - 한 블록당 128개 float를 처리
// - Warp0: cp.async로 16B씩 (float4) 로드
// - Warp1: shared에서 compute
// -------------------------------------------
__global__ void ws_kernel(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int N)
{
    extern __shared__ float smem[];
    float* bufA = smem;        // 128 floats
    float* bufB = smem + 128;  // 128 floats

    int warpId = threadIdx.x / 32;
    int lane   = threadIdx.x % 32;

    // one block handles 128 floats from A/B
    int warpTile = blockIdx.x * 128;
    if (warpTile + 128 > N) return;  // 간단 범위 체크

    // Warp0: global -> shared, cp.async (16 bytes = 4 floats)
    if (warpId == 0)
    {
        // 각 lane이 float 4개씩 담당: 32 lanes * 4 = 128 floats
        int idx = lane * 4;  // float index

        if (idx < 128)
        {
            // shared memory 주소는 32-bit 주소로 변환
            unsigned int smem_addr_A = __cvta_generic_to_shared(bufA + idx);
            unsigned int smem_addr_B = __cvta_generic_to_shared(bufB + idx);

            const float* gA = A + warpTile + idx;
            const float* gB = B + warpTile + idx;

        #if !defined(__INTELLISENSE__)
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], %2;\n" :: 
                "r"(smem_addr_A),   // 32-bit shared addr
                "l"(gA),            // 64-bit global addr
                "n"(16)             // 16 bytes = 4 floats
            );
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], %2;\n" :: 
                "r"(smem_addr_B),
                "l"(gB),
                "n"(16)
            );
        #endif
        }

        #if !defined(__INTELLISENSE__)
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        #endif
    }

    __syncthreads(); // sync loader→compute warps

    // Warp1: shared에서 compute
    if (warpId == 1)
    {
        float sum = 0.f;
        for (int i = lane; i < 128; i += 32)
        {
            float a = bufA[i];
            float b = bufB[i];
            sum += a * b;
        }
        // warp reduce
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);

        if (lane == 0)
            C[blockIdx.x] = sum;
    }
}

// -------------------------------------------
// Host driver
// -------------------------------------------
int main()
{
    int N = 1 << 20; // 1M elements
    size_t bytes = N * sizeof(float);

    float *A, *B, *C1, *C2;
    CHECK_CUDA(cudaMalloc(&A, bytes));
    CHECK_CUDA(cudaMalloc(&B, bytes));
    CHECK_CUDA(cudaMalloc(&C1, bytes));
    CHECK_CUDA(cudaMalloc(&C2, bytes));

    // init (원래 코드처럼 대충 1로 채우기)
    CHECK_CUDA(cudaMemset(A, 1, bytes));
    CHECK_CUDA(cudaMemset(B, 1, bytes));
    CHECK_CUDA(cudaMemset(C1, 0, bytes));
    CHECK_CUDA(cudaMemset(C2, 0, bytes));

    dim3 block(64);          // 2 warps
    dim3 grid(N / 128);      // one block per 128 elements

    printf("N=%d, grid=%d, block=%d\n", N, grid.x, block.x);

    // warmup
    mono_kernel<<<grid, block>>>(A, B, C1, N);
    ws_kernel<<<grid, block, 256 * sizeof(float)>>>(A, B, C2, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // mono
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; ++i)
        mono_kernel<<<grid, block>>>(A, B, C1, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float tMono;
    CHECK_CUDA(cudaEventElapsedTime(&tMono, start, stop));

    // ws
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; ++i)
        ws_kernel<<<grid, block, 256 * sizeof(float)>>>(A, B, C2, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float tWS;
    CHECK_CUDA(cudaEventElapsedTime(&tWS, start, stop));

    printf("mono: %.4f ms\n", tMono / 100.0f);
    printf("ws  : %.4f ms\n", tWS / 100.0f);
    printf("speedup: %.2fx\n", (tMono / tWS));

    CHECK_CUDA(cudaFree(A));
    CHECK_CUDA(cudaFree(B));
    CHECK_CUDA(cudaFree(C1));
    CHECK_CUDA(cudaFree(C2));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}

/*
빌드 예시 (Windows, sm_86):

nvcc warp_specialization_test.cu -o warp_specialization_test.exe      -arch=sm_86 -O3 --use_fast_math --ptxas-options=-v

ncu   --kernel-name regex:mono_kernel.*   --launch-skip 5   --launch-count 1   --set full   .\warp_specialization_test.exe
ncu   --kernel-name regex:ws_kernel.*   --launch-skip 5   --launch-count 1   --set full   .\warp_specialization_test.exe

*/

