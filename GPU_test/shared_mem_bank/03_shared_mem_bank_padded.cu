#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err__ = (call);                           \
        if (err__ != cudaSuccess) {                           \
            std::cerr << "CUDA error: "                       \
                      << cudaGetErrorString(err__)            \
                      << " at " << __FILE__ << ":" << __LINE__\
                      << std::endl;                           \
            std::exit(1);                                     \
        }                                                     \
    } while (0)

// -----------------------------------------------------------------------------
// case A: padding 없이, stride=32 → 최악 32-way bank conflict
// shared[tid * 32]  → bank = (tid * 32) % 32 = 0
// -----------------------------------------------------------------------------
__global__ void shared_bank_nopad(float* out, int iters, int stride)
{
    __shared__ volatile float sh[32 * 32];  // 1024 floats

    int tid = threadIdx.x;
    if (tid >= 32) return; // warp 0만 사용

    sh[tid] = tid;
    __syncthreads();

    float acc = 0.0f;

    for (int i = 0; i < iters; ++i) {
        // stride=32일 때: idx = tid*32 → bank = 0 (최악)
        int idx = (tid * stride) & (32 * 32 - 1);
        acc += sh[idx];
    }

    out[tid] = acc;
}

// -----------------------------------------------------------------------------
// case B: padding 사용 (32×33 배열)
// 접근: idx = tid * 33  → bank = (tid * 33) % 32 = tid
// → warp 32 threads → bank 32개에 1:1 매핑 (conflict-free)
// -----------------------------------------------------------------------------
__global__ void shared_bank_padded(float* out, int iters)
{
    __shared__ volatile float sh[32 * 33];  // padding 컬럼 추가 (33)

    int tid = threadIdx.x;
    if (tid >= 32) return;

    sh[tid * 33] = tid;  // 각 thread가 자기 "행"에 초기화
    __syncthreads();

    float acc = 0.0f;

    for (int i = 0; i < iters; ++i) {
        int idx = (tid * 33) & (32 * 33 - 1);
        acc += sh[idx];
    }

    out[tid] = acc;
}

// 공통 타이머 유틸
float run_nopad(float* d_out, int iters, int stride)
{
    const int block_size = 256;
    const int grid_size  = 1;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warmup
    shared_bank_nopad<<<grid_size, block_size>>>(d_out, iters, stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    shared_bank_nopad<<<grid_size, block_size>>>(d_out, iters, stride);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

float run_padded(float* d_out, int iters)
{
    const int block_size = 256;
    const int grid_size  = 1;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warmup
    shared_bank_padded<<<grid_size, block_size>>>(d_out, iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    shared_bank_padded<<<grid_size, block_size>>>(d_out, iters);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

int main()
{
    const int iters = 1'000'000;
    const int out_size = 32;

    std::cout << "Shared memory bank conflict (padding) test\n";
    std::cout << "iters = " << iters << "\n\n";

    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, out_size * sizeof(float)));
    std::vector<float> h_out(out_size);

    // case A: padding 없음, stride=32 (최악)
    const int stride = 32;
    float ms_nopad = run_nopad(d_out, iters, stride);

    // case B: padding 있음
    float ms_padded = run_padded(d_out, iters);

    // 결과 읽기 (최적화 방지용)
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                          out_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_out));

    std::cout << "No padding (stride=32, worst case): " << std::fixed << std::setprecision(4)
              << ms_nopad << " ms\n";
    std::cout << "With padding (33 pitch, conflict-free): " << std::fixed << std::setprecision(4)
              << ms_padded << " ms\n";

    std::cout << "Speedup (nopad / padded) ~= " << (ms_nopad / ms_padded) << "x\n";

    return 0;
}

// nvcc .\04_shared_mem_bank_padded.cu -o .\04_shared_mem_bank_padded.exe