#include <iostream>
#include <cuda_runtime.h>

#define N (1024 * 1024 * 10)  // 10M 요소
#define THREADS_PER_BLOCK 256
#define REPEATS 1000

__global__ void coalesced_read(float* input, float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        output[idx] = input[idx];
}

__global__ void non_coalesced_read(float* input, float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        output[idx] = input[(idx * 128) % N];  // 큰 stride로 non-coalesced 접근
}

template<typename Kernel>
void benchmark(const char* label, Kernel kernel, float* input, float* output) {
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEventRecord(start);
    for (int i = 0; i < REPEATS; ++i) {
        kernel<<<blocks, THREADS_PER_BLOCK>>>(input, output);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, end);

    // 최적화 방지를 위한 출력 체크
    float* host_output = new float[10];
    cudaMemcpy(host_output, output, sizeof(float) * 10, cudaMemcpyDeviceToHost);
    float checksum = 0;
    for (int i = 0; i < 10; ++i)
        checksum += host_output[i];
    delete[] host_output;

    std::cout << label << " Time over " << REPEATS << " runs: " << elapsed << " ms, Checksum: " << checksum << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(end);
}

int main() {
    float *input, *output;
    cudaMalloc(&input, N * sizeof(float));
    cudaMalloc(&output, N * sizeof(float));

    // 간단한 초기화 (모든 값 1.0f)
    float* host_input = new float[N];
    for (int i = 0; i < N; ++i) host_input[i] = 1.0f;
    cudaMemcpy(input, host_input, N * sizeof(float), cudaMemcpyHostToDevice);
    delete[] host_input;

    benchmark("Coalesced", coalesced_read, input, output);
    benchmark("Non-Coalesced", non_coalesced_read, input, output);

    cudaFree(input);
    cudaFree(output);
    return 0;
}
