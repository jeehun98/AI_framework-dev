#include <cuda_runtime.h>
#include <iostream>
#include "run_graph.cuh"

#define TILE_WIDTH 16

__global__ void matmul_shared_row_major(float* A, float* B, float* C,
                                        int A_rows, int A_cols, int B_cols) {
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float val = 0.0f;

    for (int t = 0; t < (A_cols + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        if (row < A_rows && t * TILE_WIDTH + threadIdx.x < A_cols)
            tile_A[threadIdx.y][threadIdx.x] = A[row * A_cols + t * TILE_WIDTH + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (t * TILE_WIDTH + threadIdx.y < A_cols && col < B_cols)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * B_cols + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i)
            val += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];

        __syncthreads();
    }

    if (row < A_rows && col < B_cols)
        C[row * B_cols + col] = val;
}

__global__ void add_bias_relu(float* input, float* bias, float* output,
                              int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        float val = input[idx] + bias[col];
        output[idx] = val > 0 ? val : 0;
    }
}

extern "C" void run_graph_cuda(int* E, int E_len, int* shapes, int shapes_len,
                               float* W, float* b, int W_rows, int W_cols) {
    int batch = shapes[1];
    int input_dim = shapes[2];

    float* x_d;
    float* W_d;
    float* b_d;
    float* out_d;

    size_t input_size = batch * input_dim * sizeof(float);
    size_t W_size = W_rows * W_cols * sizeof(float);
    size_t b_size = W_cols * sizeof(float);
    size_t out_size = batch * W_cols * sizeof(float);

    float* x = new float[batch * input_dim];
    for (int i = 0; i < batch * input_dim; ++i)
        x[i] = 1.0f;

    cudaMalloc(&x_d, input_size);
    cudaMalloc(&W_d, W_size);
    cudaMalloc(&b_d, b_size);
    cudaMalloc(&out_d, out_size);

    cudaMemcpy(x_d, x, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(W_d, W, W_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, b_size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((W_cols + TILE_WIDTH - 1) / TILE_WIDTH,
                 (batch + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_shared_row_major<<<dimGrid, dimBlock>>>(x_d, W_d, out_d, batch, input_dim, W_cols);

    int total = batch * W_cols;
    add_bias_relu<<<(total + 255) / 256, 256>>>(out_d, b_d, out_d, batch, W_cols);

    float* out_host = new float[batch * W_cols];
    cudaMemcpy(out_host, out_d, out_size, cudaMemcpyDeviceToHost);

    std::cout << "✅ CUDA 그래프 실행 완료!\n출력: ";
    for (int i = 0; i < batch * W_cols; ++i) {
        std::cout << out_host[i] << " ";
    }
    std::cout << std::endl;

    delete[] out_host;
    delete[] x;
    cudaFree(x_d);
    cudaFree(W_d);
    cudaFree(b_d);
    cudaFree(out_d);
}
