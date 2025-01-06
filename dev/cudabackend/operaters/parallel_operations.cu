#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// CUDA 커널: 입력 배열의 각 요소에 제곱 연산 수행
__global__ void square_kernel(float* input, float* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        output[idx] = input[idx] * input[idx];
    }
}

// CUDA 실행 함수: Pybind11에서 호출 가능
void square(const std::vector<float>& input, std::vector<float>& output) {
    int n = input.size();
    float *d_input, *d_output;

    // GPU 메모리 할당
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    // 데이터 복사 (Host to Device)
    cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // 블록 및 그리드 크기 설정
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // 커널 호출
    square_kernel<<<gridSize, blockSize>>>(d_input, d_output, n);

    // 결과 복사 (Device to Host)
    cudaMemcpy(output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // GPU 메모리 해제
    cudaFree(d_input);
    cudaFree(d_output);
}
