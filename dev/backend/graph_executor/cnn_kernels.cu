#include <cuda_runtime.h>
#include "cnn_kernels.cuh"

#define TILE_WIDTH 16

// ✅ Forward: Convolution Layer (single-channel, no padding, stride=1)
__global__ void conv2d_forward_kernel(const float* input, const float* kernel,
                                      float* output, int B, int H, int W,
                                      int KH, int KW, int OH, int OW) {
    int b = blockIdx.z;  // batch index
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // output row
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // output col

    if (row < OH && col < OW) {
        float sum = 0.0f;
        for (int i = 0; i < KH; ++i) {
            for (int j = 0; j < KW; ++j) {
                int in_row = row + i;
                int in_col = col + j;
                sum += input[b * H * W + in_row * W + in_col] *
                       kernel[i * KW + j];
            }
        }
        output[b * OH * OW + row * OW + col] = sum;
    }
}

// ✅ Backward: dL/dInput (for input propagation)
__global__ void conv2d_backward_input_kernel(const float* d_out, const float* kernel,
                                             float* d_input, int B, int H, int W,
                                             int KH, int KW, int OH, int OW) {
    int b = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < H && col < W) {
        float sum = 0.0f;
        for (int i = 0; i < KH; ++i) {
            for (int j = 0; j < KW; ++j) {
                int out_row = row - i;
                int out_col = col - j;
                if (out_row >= 0 && out_row < OH && out_col >= 0 && out_col < OW) {
                    sum += d_out[b * OH * OW + out_row * OW + out_col] *
                           kernel[i * KW + j];
                }
            }
        }
        d_input[b * H * W + row * W + col] = sum;
    }
}

// ✅ Backward: dL/dKernel
__global__ void conv2d_backward_kernel_kernel(const float* input, const float* d_out,
                                              float* d_kernel, int B, int H, int W,
                                              int KH, int KW, int OH, int OW) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < KH && j < KW) {
        float sum = 0.0f;
        for (int b = 0; b < B; ++b) {
            for (int row = 0; row < OH; ++row) {
                for (int col = 0; col < OW; ++col) {
                    int in_row = row + i;
                    int in_col = col + j;
                    sum += input[b * H * W + in_row * W + in_col] *
                           d_out[b * OH * OW + row * OW + col];
                }
            }
        }
        d_kernel[i * KW + j] = sum;
    }
}
