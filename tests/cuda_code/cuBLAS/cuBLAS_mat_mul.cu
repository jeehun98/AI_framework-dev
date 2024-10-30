#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 3  // 행렬의 크기

int main() {
    // 호스트에서 사용하는 행렬
    float h_A[N * N] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float h_B[N * N] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    float h_C[N * N] = {0};  // 결과 행렬

    // GPU 메모리 할당
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    // 메모리 복사 (호스트 -> 디바이스)
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // cuBLAS 핸들 생성
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 행렬 곱셈 파라미터 설정
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 행렬 곱셈: C = alpha * A * B + beta * C
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

    // 결과를 호스트로 복사
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 결과 출력
    std::cout << "Result Matrix C:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // 리소스 해제
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
