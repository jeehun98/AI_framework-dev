#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void multiplyByTwoGPU(int *a, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) {
        a[idx] *= 2;
    }
}

void multiplyByTwoCPU(int *a, int size) {
    for (int i = 0; i < size; i++) {
        a[i] *= 2;
    }
}

int main() {
    const int size = 1000000; // 데이터 크기 증가
    int *h_a = new int[size];
    for (int i = 0; i < size; i++) {
        h_a[i] = i;
    }

    // GPU 연산 시간 측정
    int *d_a;
    cudaMalloc(&d_a, size * sizeof(int));
    cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    multiplyByTwoGPU<<<numBlocks, blockSize>>>(d_a, size);
    cudaDeviceSynchronize();  // 커널 실행이 끝날 때까지 대기
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> gpu_duration = end_gpu - start_gpu;

    cudaMemcpy(h_a, d_a, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_a);

    // CPU 연산 시간 측정
    auto start_cpu = std::chrono::high_resolution_clock::now();
    multiplyByTwoCPU(h_a, size);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = end_cpu - start_cpu;

    // 결과 출력
    std::cout << "GPU 실행 시간: " << gpu_duration.count() << " ms" << std::endl;
    std::cout << "CPU 실행 시간: " << cpu_duration.count() << " ms" << std::endl;

    delete[] h_a;
    return 0;
}
