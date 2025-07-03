#include <iostream>
#include <cuda_runtime.h>
#include "run_graph.cuh"
#include "matmul_shared_optimized.cuh"
#include "activation.cuh"

#define TILE_WIDTH 16

void run_graph_cuda(int* E, int E_len, int* shapes, int shapes_len,
                    float* W, float* b, int W_rows, int W_cols,
                    int activation_type, float* out_host) {
    int batch = shapes[1];
    int input_dim = shapes[2];

    float *x_d, *W_d, *b_d, *out_d;
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
    dim3 dimGrid((W_cols + TILE_WIDTH - 1) / TILE_WIDTH, (batch + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_shared_kernel_coalesced<<<dimGrid, dimBlock>>>(x_d, W_d, out_d, batch, input_dim, W_cols);

    int total = batch * W_cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    // ✅ 커널 직접 launch
    switch (activation_type) {
        case ACT_RELU:
            activation_relu<<<blocks, threads>>>(out_d, b_d, out_d, batch, W_cols);
            break;
        case ACT_SIGMOID:
            activation_sigmoid<<<blocks, threads>>>(out_d, b_d, out_d, batch, W_cols);
            break;
        case ACT_TANH:
            activation_tanh<<<blocks, threads>>>(out_d, b_d, out_d, batch, W_cols);
            break;
        default:
            printf("Unsupported activation type: %d\n", activation_type);
            break;
    }

    cudaMemcpy(out_host, out_d, out_size, cudaMemcpyDeviceToHost);

    delete[] x;
    cudaFree(x_d);
    cudaFree(W_d);
    cudaFree(b_d);
    cudaFree(out_d);
}
