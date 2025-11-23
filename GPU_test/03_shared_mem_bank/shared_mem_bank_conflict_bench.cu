// shared_mem_bank_conflict_bench.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include <cstdlib>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err__ = (call); \
        if (err__ != cudaSuccess) { \
            std::cerr << "CUDA error: " \
                      << cudaGetErrorString(err__) \
                      << " at " << __FILE__ << ":" << __LINE__ \
                      << std::endl; \
            std::exit(1); \
        } \
    } while (0)


// -----------------------------------------------------------------------------
// Shared memory bank conflict kernel
//   - warp 32 threads만 사용 (tid 0~31)
//   - stride 변화로 conflict 정도 비교
// -----------------------------------------------------------------------------
__global__ void shared_bank_kernel(float* out, int iters, int stride)
{
    __shared__ volatile float sh[32 * 32];   // 1024 floats

    int tid = threadIdx.x;
    // 단일 워프만 (32개 스레드)
    if (tid >= 32) return;

    sh[tid] = tid;
    __syncthreads();

    float acc = 0.0f;

    for (int i = 0; i < iters; ++i) {
        int idx = (tid * stride) & (32 * 32 - 1);
        acc += sh[idx];
    }

    out[tid] = acc;
}


// -----------------------------------------------------------------------------
// 실행 모드별 커널 런 타이밍 함수
// -----------------------------------------------------------------------------
float run_bank_test(float* d_out, int iters, int stride)
{
    // 단일 블록 256 스레드
    const int block_size = 256;
    const int grid_size  = 1;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    shared_bank_kernel<<<grid_size, block_size>>>(d_out, iters, stride);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    shared_bank_kernel<<<grid_size, block_size>>>(d_out, iters, stride);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}


// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string mode = "bench";  // bench | profile
    if (argc >= 2) mode = argv[1];

    int iters = 1'000'000;  // default

    if (mode == "profile") {
        // Nsight Compute 프로파일용 : 짧아야 타임라인 보기 쉽다
        iters = 20;
    }

    const int out_size = 32;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, out_size * sizeof(float)));

    std::vector<float> h_out(out_size);

    std::cout << "Shared Memory Bank Conflict Experiment (" << mode << ")\n";
    std::cout << "iters = " << iters << "\n\n";

    // 실험할 stride 목록 (확장 버전)
    int strides[] = {
        1, 2, 4, 8, 16, 32,
        33, 34, 40, 64
    };
    int num = sizeof(strides) / sizeof(strides[0]);

    std::cout << "stride | time (ms)\n";
    std::cout << "------------------\n";

    for (int s = 0; s < num; ++s) {
        float ms = run_bank_test(d_out, iters, strides[s]);
        std::cout << std::setw(6) << strides[s]
                  << " | " << std::fixed << std::setprecision(6)
                  << ms << "\n";
    }

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                          out_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_out));
    return 0;
}

/*
빌드:
    nvcc -O3 -lineinfo shared_mem_bank_conflict_bench.cu -o bank_bench.exe

Bench 실행:
    ./bank_bench.exe bench

Nsight Compute 프로파일 (짧은 반복):
    ncu -o bank_profile --set full ./bank_bench.exe profile
*/
