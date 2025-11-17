#include <cuda_runtime.h>
#include <iostream>
#include <vector>

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

__global__ void coalesced_read(const float* __restrict__ in,
                               float* __restrict__ out,
                               int N)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    // 완전 연속 접근 (coalesced)
    out[gid] = in[gid];
}

__global__ void strided_read(const float* __restrict__ in,
                             float* __restrict__ out,
                             int N,
                             int stride)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    // strided 접근
    int idx = (gid * stride) % N;  // 일부러 뒤섞기
    out[gid] = in[idx];
}

float run_kernel_coalesced(const float* d_in, float* d_out, int N,
                           int block_size, int iters)
{
    int grid_size = (N + block_size - 1) / block_size;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warmup
    coalesced_read<<<grid_size, block_size>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        coalesced_read<<<grid_size, block_size>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // iters 번 돌린 총 시간(ms)
    return ms;
}

float run_kernel_strided(const float* d_in, float* d_out, int N,
                         int block_size, int stride, int iters)
{
    int grid_size = (N + block_size - 1) / block_size;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warmup
    strided_read<<<grid_size, block_size>>>(d_in, d_out, N, stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        strided_read<<<grid_size, block_size>>>(d_in, d_out, N, stride);
    }
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
    const int N = 1 << 24;      // 대략 16M float (~64MB)
    const int block_size = 256;
    const int iters = 50;

    std::cout << "N=" << N
              << ", block_size=" << block_size
              << ", iters=" << iters << std::endl;

    // host 버퍼
    std::vector<float> h_in(N, 1.0f);
    std::vector<float> h_out(N, 0.0f);

    // device 버퍼
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));

    float ms_coalesced = run_kernel_coalesced(d_in, d_out, N,
                                              block_size, iters);
    std::cout << "[coalesced] total time: " << ms_coalesced
              << " ms, per iter: " << (ms_coalesced / iters) << " ms"
              << std::endl;

    for (int stride : {2, 4, 8, 16, 32}) {
        float ms_strided = run_kernel_strided(d_in, d_out, N,
                                              block_size, stride, iters);
        std::cout << "[stride=" << stride << "] total time: " << ms_strided
                  << " ms, per iter: " << (ms_strided / iters) << " ms"
                  << std::endl;
    }

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}

//nvcc .\02_coalescing.cu -o .\02_coalescing.exe .\02_coalescing.exe

//ncu -o 02_coalescing_report --set full .\02_coalescing.exe
