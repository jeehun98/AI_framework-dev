#include <cuda_runtime.h>
#include "run_graph.cuh"
#include <stdio.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum += A[row * N + i] * B[i * K + col];
    }
    C[row * K + col] = sum;
}

__global__ void add_bias_kernel(float* C, const float* b, int M, int K) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    C[row * K + col] += b[col];
}

__global__ void relu_kernel(float* C, int size) {
    int idx = threadIdx.x;
    if (idx < size) {
        if (C[idx] < 0) C[idx] = 0;
    }
}

extern "C" void run_graph_cuda(
    int* E, int E_len,
    int* shapes, int shapes_len,
    float* W, float* b,
    int W_rows, int W_cols,
    float* x, float* out
) {
    float *x_d, *W_d, *b_d, *out_d;
    cudaMalloc((void**)&x_d, sizeof(float) * W_rows);
    cudaMalloc((void**)&W_d, sizeof(float) * W_rows * W_cols);
    cudaMalloc((void**)&b_d, sizeof(float) * W_cols);
    cudaMalloc((void**)&out_d, sizeof(float) * W_cols);

    cudaMemcpy(x_d, x, sizeof(float) * W_rows, cudaMemcpyHostToDevice);
    cudaMemcpy(W_d, W, sizeof(float) * W_rows * W_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, sizeof(float) * W_cols, cudaMemcpyHostToDevice);

    dim3 blockDim(W_cols, 1);
    matmul_kernel<<<1, blockDim>>>(x_d, W_d, out_d, 1, W_rows, W_cols);

    add_bias_kernel<<<1, blockDim>>>(out_d, b_d, 1, W_cols);
    relu_kernel<<<1, W_cols>>>(out_d, W_cols);

    cudaMemcpy(out, out_d, sizeof(float) * W_cols, cudaMemcpyDeviceToHost);

    cudaFree(x_d);
    cudaFree(W_d);
    cudaFree(b_d);
    cudaFree(out_d);
}
