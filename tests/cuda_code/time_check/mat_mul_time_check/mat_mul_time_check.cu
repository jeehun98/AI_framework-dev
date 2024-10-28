#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define N 512  // 행렬 크기 (N x N)

// GPU에서 실행되는 커널 함수
__global__ void matrixMultiply(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// CPU에서 실행되는 행렬 곱셈 함수 (비교용)
void matrixMultiplyCPU(const float *A, const float *B, float *C, int n) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            float sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[row * n + k] * B[k * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}

int main() {
    int size = N * N;
    int bytes = size * sizeof(float);

    // 호스트 메모리 할당
    float *h_A = new float[size];
    float *h_B = new float[size];
    float *h_C = new float[size];
    float *h_C_CPU = new float[size];

    // 행렬 초기화
    for (int i = 0; i < size; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    // GPU 메모리 할당
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // 호스트에서 GPU로 데이터 복사
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 블록과 그리드 차원 설정
    dim3 blockSize(16, 16);  // 16x16 스레드
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // GPU에서 행렬 곱 연산 및 시간 측정
    auto start_gpu = std::chrono::high_resolution_clock::now();
    matrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();  // 커널이 완료될 때까지 대기
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> gpu_duration = end_gpu - start_gpu;

    // GPU에서 호스트로 결과 복사
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // CPU에서 행렬 곱 연산 및 시간 측정
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matrixMultiplyCPU(h_A, h_B, h_C_CPU, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = end_cpu - start_cpu;

    // 결과 비교
    bool correct = true;
    for (int i = 0; i < size; i++) {
        if (abs(h_C[i] - h_C_CPU[i]) > 1e-4) {
            correct = false;
            break;
        }
    }

    // 결과 출력
    if (correct) {
        std::cout << "결과가 정확합니다." << std::endl;
    } else {
        std::cout << "결과가 정확하지 않습니다." << std::endl;
    }

    std::cout << "GPU time: " << gpu_duration.count() << " ms" << std::endl;
    std::cout << "CPU time: " << cpu_duration.count() << " ms" << std::endl;

    // 메모리 해제
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_CPU;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
