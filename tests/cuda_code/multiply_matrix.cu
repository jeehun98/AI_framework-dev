#include <stdio.h>

// 행렬 크기 정의
#define N 16

// CUDA 커널 함수: 각 스레드가 행렬 C의 원소 하나를 계산
__global__ void matrixMul(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;

    // row, col이 유효한 범위 내에 있을 때에만 계산 수행
    if (row < n && col < n) {
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    int n = N;
    size_t size = n * n * sizeof(float);
    
    // 호스트(Host) 메모리 할당
    float h_A[N*N], h_B[N*N], h_C[N*N];
    
    // 초기화 (임의의 값을 사용)
    for (int i = 0; i < N*N; i++) {
        h_A[i] = 1.0f;  // 간단한 테스트용 값
        h_B[i] = 2.0f;
    }
    
    // 장치(Device) 메모리 할당
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    
    // 호스트 메모리에서 장치 메모리로 데이터 복사
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // 블록 및 그리드 크기 정의
    dim3 threadsPerBlock(16, 16);  // 각 블록 내의 스레드 수 (16x16)
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // CUDA 커널 호출
    matrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
    
    // 장치 메모리에서 호스트 메모리로 결과 복사
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // 결과 출력
    printf("Result matrix C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // 장치 메모리 해제
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
