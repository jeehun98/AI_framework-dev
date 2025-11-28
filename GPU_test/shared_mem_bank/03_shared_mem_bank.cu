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
// shared memory bank conflict 실험용 커널
//  - warp 1개(32 threads)만 사용
//  - stride를 바꾸면서 shared 메모리에서 반복 접근
//  - stride에 따라 bank conflict 정도가 달라짐
// -----------------------------------------------------------------------------
__global__ void shared_bank_kernel(float* out, int iters, int stride)
{
    // 32개의 bank * 여러 라인
    __shared__ volatile float sh[32 * 32];

    int tid = threadIdx.x;

    // warp 0만 사용 (0~31)
    if (tid >= 32) return;

    // shared mem 초기화 (값은 아무거나 상관없음)
    // 모든 thread가 같은 인덱스를 쓰지만, 여기서는 성능 중요 X
    sh[tid] = tid;

    __syncthreads();

    float acc = 0.0f;

    // bank index = (address / 4) % 32 ≈ (index % 32)
    // index = tid * stride 일 때,
    //   bank = (tid * stride) % 32
    //   stride 값에 따라 bank conflict 패턴이 달라짐
    for (int i = 0; i < iters; ++i) {
        int idx = (tid * stride) & (32 * 32 - 1); // 배열 크기 안으로 마스킹
        acc += sh[idx];
    }

    // 결과를 global mem으로 내보내서 최적화 방지
    out[tid] = acc;
}

// 특정 stride 에 대해 kernel 시간 측정
float run_bank_test(float* d_out, int iters, int stride)
{
    const int block_size = 256;
    const int grid_size  = 1;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warmup
    shared_bank_kernel<<<grid_size, block_size>>>(d_out, iters, stride);
    CUDA_CHECK(cudaGetLastError());
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

int main()
{
    const int iters = 1'000'000;   // 반복 횟수 (필요하면 줄여도 됨)
    const int out_size = 32;

    std::cout << "Shared memory bank conflict test\n";
    std::cout << "iters = " << iters << "\n";

    // device output
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, out_size * sizeof(float)));

    // host buffer
    std::vector<float> h_out(out_size);

    // 테스트할 stride 값들
    int strides[] = {1, 2, 4, 8, 16, 32};

    std::cout << "stride | time (ms)\n";
    std::cout << "-----------------\n";

    for (int s : strides) {
        float ms = run_bank_test(d_out, iters, s);
        std::cout << std::setw(6) << s << " | "
                  << std::fixed << std::setprecision(4)
                  << ms << "\n";
    }

    // 한 번 읽어와서 그냥 쓰레기라도 써보기 (최적화 방지 겸)
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                          out_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_out));
    return 0;
}

//nvcc .\03_shared_mem_bank.cu -o .\03_shared_mem_bank.exe 
