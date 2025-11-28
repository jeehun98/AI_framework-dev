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
// 공통: 간단한 dummy 연산
// -----------------------------------------------------------------------------
__device__ float do_work(float x, int iters)
{
    // 약간의 floating 연산으로 loop를 무겁게
    for (int i = 0; i < iters; ++i) {
        x = x * 1.000001f + 1.0f;
        x = x - 1.0f;
    }
    return x;
}

// -----------------------------------------------------------------------------
// case A: 분기 없음 (warp divergence 없음)
// 모든 thread가 동일 경로로 같은 연산 수행
// -----------------------------------------------------------------------------
__global__ void kernel_no_divergence(float* out, int iters)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float x = static_cast<float>(gid);

    x = do_work(x, iters);

    out[gid] = x;
}

// -----------------------------------------------------------------------------
// case B: warp 내부에서 절반은 if, 절반은 else
//  - threadIdx.x % 2 == 0 → branch A
//  - threadIdx.x % 2 == 1 → branch B
// 두 branch 모두 동일 연산량(do_work)을 수행
// → 순수하게 divergence만 성능에 미치는 영향 보기
// -----------------------------------------------------------------------------
__global__ void kernel_half_divergence(float* out, int iters)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float x = static_cast<float>(gid);

    if ((threadIdx.x & 1) == 0) {
        // 짝수 thread
        x = do_work(x, iters);
    } else {
        // 홀수 thread
        x = do_work(x, iters);
    }

    out[gid] = x;
}

// -----------------------------------------------------------------------------
// case C: 극단적 divergence (threadIdx.x % 32 기준)
// warp 내에서:
//   lane 0 → 무거운 연산
//   lane 1~31 → 가벼운 연산
// warp 관점에서 보면, "무거운 한 스레드 때문에 warp 전체가 잡혀 있는" 상황
// -----------------------------------------------------------------------------
__global__ void kernel_extreme_divergence(float* out, int heavy_iters, int light_iters)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float x = static_cast<float>(gid);

    int lane = threadIdx.x & 31; // warp lane

    if (lane == 0) {
        // warp당 1 thread만 매우 무거운 일
        x = do_work(x, heavy_iters);
    } else {
        // 나머지는 가벼운 일
        x = do_work(x, light_iters);
    }

    out[gid] = x;
}

// -----------------------------------------------------------------------------
// 타이머 유틸
// -----------------------------------------------------------------------------
float run_kernel_no_div(float* d_out, int num_threads, int iters)
{
    const int block_size = 256;
    int grid_size = (num_threads + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warmup
    kernel_no_divergence<<<grid_size, block_size>>>(d_out, iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    kernel_no_divergence<<<grid_size, block_size>>>(d_out, iters);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

float run_kernel_half_div(float* d_out, int num_threads, int iters)
{
    const int block_size = 256;
    int grid_size = (num_threads + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warmup
    kernel_half_divergence<<<grid_size, block_size>>>(d_out, iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    kernel_half_divergence<<<grid_size, block_size>>>(d_out, iters);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

float run_kernel_extreme_div(float* d_out, int num_threads,
                             int heavy_iters, int light_iters)
{
    const int block_size = 256;
    int grid_size = (num_threads + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warmup
    kernel_extreme_divergence<<<grid_size, block_size>>>(d_out, heavy_iters, light_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    kernel_extreme_divergence<<<grid_size, block_size>>>(d_out, heavy_iters, light_iters);
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
    // 전체 스레드 수 (적당히 크게)
    const int num_threads = 1 << 20;   // ~1M threads
    const int iters = 2000;            // per-thread work 반복 횟수
    const int heavy_iters = 2000;
    const int light_iters = 10;

    std::cout << "Warp divergence test\n";
    std::cout << "num_threads = " << num_threads
              << ", iters = " << iters << "\n\n";

    // device output
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, num_threads * sizeof(float)));
    std::vector<float> h_out(num_threads);

    float t_no = run_kernel_no_div(d_out, num_threads, iters);
    float t_half = run_kernel_half_div(d_out, num_threads, iters);
    float t_extreme = run_kernel_extreme_div(d_out, num_threads,
                                             heavy_iters, light_iters);

    // 결과 한 번 읽어와서 최적화 방지
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                          num_threads * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "[no divergence]          : " << t_no      << " ms\n";
    std::cout << "[half divergence]        : " << t_half    << " ms\n";
    std::cout << "[extreme divergence]     : " << t_extreme << " ms\n";

    std::cout << "half / no       ≈ " << (t_half / t_no) << "x\n";
    std::cout << "extreme / no    ≈ " << (t_extreme / t_no) << "x\n";

    return 0;
}

// nvcc .\04_warp_divergence.cu -o .\04_warp_divergence.exe
