// global_memory_coalescing_bench.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err__ = (call);                                     \
        if (err__ != cudaSuccess) {                                     \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)    \
                      << " at " << __FILE__ << ":" << __LINE__          \
                      << std::endl;                                     \
            std::exit(1);                                               \
        }                                                               \
    } while (0)

// -----------------------------------------------------------------------------
// Kernels: global memory coalesced vs strided
// -----------------------------------------------------------------------------

__global__ void coalesced_read_kernel(const float* __restrict__ in,
                                      float* __restrict__ out,
                                      int N)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    // 완전 연속(global coalesced) 접근
    float x = in[gid];
    out[gid] = x;
}

__global__ void strided_read_kernel(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int N,
                                    int stride)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    // global 에서 strided 접근
    int idx = (gid * stride) % N;
    float x = in[idx];
    out[gid] = x;
}

// -----------------------------------------------------------------------------
// Timing helpers
// -----------------------------------------------------------------------------

float run_coalesced(const float* d_in, float* d_out,
                    int N, int block_size, int iters)
{
    int grid_size = (N + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warm-up
    coalesced_read_kernel<<<grid_size, block_size>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        coalesced_read_kernel<<<grid_size, block_size>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;  // total ms over iters
}

float run_strided(const float* d_in, float* d_out,
                  int N, int block_size, int stride, int iters)
{
    int grid_size = (N + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warm-up
    strided_read_kernel<<<grid_size, block_size>>>(d_in, d_out, N, stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        strided_read_kernel<<<grid_size, block_size>>>(d_in, d_out, N, stride);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;  // total ms over iters
}

// -----------------------------------------------------------------------------
// Main
//   실행 인자:
//     argv[1] : mode      ("bench" / "profile")
//     argv[2] : N
//     argv[3] : block_size
//     argv[4] : iters
// -----------------------------------------------------------------------------

int main(int argc, char** argv)
{
    // 기본값 (bench 기준)
    int N          = 1 << 24;   // ~16M floats (~64MB)
    int block_size = 256;
    int iters      = 50;
    std::string mode = "bench";

    if (argc >= 2) mode       = argv[1];
    if (argc >= 3) N          = std::atoi(argv[2]);
    if (argc >= 4) block_size = std::atoi(argv[3]);
    if (argc >= 5) iters      = std::atoi(argv[4]);

    if (mode != "bench" && mode != "profile") {
        std::cerr << "Unknown mode: " << mode
                  << " (use \"bench\" or \"profile\")\n";
        mode = "bench";
    }

    // Nsight 용 profile 모드: 반복 최소화, 필요하면 N 줄이기
    if (mode == "profile") {
        if (iters > 3) iters = 3;
        // 필요하면 N도 줄이고 싶으면 이 부분 풀면 됨
        // if (N > (1 << 22)) N = 1 << 22;
    }

    // 디바이스 정보
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "=== Global Memory Coalescing (" << mode << " mode) ===\n";
    std::cout << "Device      : " << prop.name << "\n";
    std::cout << "N           : " << N << "\n";
    std::cout << "block_size  : " << block_size << "\n";
    std::cout << "iters       : " << iters << "\n";

    double bytes_per_iter = 2.0 * static_cast<double>(N) * sizeof(float); // read + write
    std::cout << "Bytes/iter  : "
              << (bytes_per_iter / (1024.0 * 1024.0)) << " MB\n\n";

    // host / device 버퍼 할당
    std::vector<float> h_in(N, 1.0f);
    std::vector<float> h_out(N, 0.0f);

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // 1) Coalesced
    // -------------------------------------------------------------------------
    float ms_coalesced = run_coalesced(d_in, d_out, N, block_size, iters);
    double total_bytes = bytes_per_iter * static_cast<double>(iters);
    double total_seconds = ms_coalesced * 1e-3;
    double gbps = (total_bytes / total_seconds) / 1e9;

    std::cout << "[coalesced] "
              << "total: " << ms_coalesced << " ms"
              << ", per_iter: " << (ms_coalesced / iters) << " ms"
              << ", BW: " << gbps << " GB/s\n";

    // 결과를 읽어서 최적화 방지
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float),
                          cudaMemcpyDeviceToHost));
    volatile float sink = h_out[0];
    (void)sink;

    // -------------------------------------------------------------------------
    // 2) 여러 stride 패턴
    // -------------------------------------------------------------------------
    int strides[] = {2, 4, 8, 16, 32, 64};
    int num_strides = static_cast<int>(sizeof(strides) / sizeof(strides[0]));

    for (int i = 0; i < num_strides; ++i) {
        int stride = strides[i];
        float ms_strided = run_strided(d_in, d_out, N,
                                       block_size, stride, iters);

        double total_bytes_s = bytes_per_iter * static_cast<double>(iters);
        double total_seconds_s = ms_strided * 1e-3;
        double gbps_s = (total_bytes_s / total_seconds_s) / 1e9;

        std::cout << "[stride=" << stride << "] "
                  << "total: " << ms_strided << " ms"
                  << ", per_iter: " << (ms_strided / iters) << " ms"
                  << ", BW: " << gbps_s << " GB/s\n";

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float),
                              cudaMemcpyDeviceToHost));
        volatile float sink2 = h_out[(i + 1) % N];
        (void)sink2;
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}

// nvcc -O3 -lineinfo global_memory_coalescing_bench.cu -o global_mem_bench.exe
// ./global_mem_bench.exe bench 16777216 256 50

// ncu -o global_mem_bench_ncu --set full ./global_mem_bench.exe profile 16777216 256 2