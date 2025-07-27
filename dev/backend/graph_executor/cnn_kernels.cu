#include <cuda_runtime.h>
#include "cnn_kernels.cuh"

#define TILE_WIDTH 16

// ✅ Forward: Convolution Layer (single-channel, no padding, stride=1)
__global__ void conv2d_forward_kernel(const float* input, const float* kernel,
                                      float* output, int B, int H, int W,
                                      int IC, int OC, int KH, int KW, int OH, int OW) {
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
__global__ void conv2d_backward_input_kernel(const float* d_out, const float* W,
                                             float* d_input, int B, int H, int W_in,
                                             int IC, int OC, int KH, int KW,
                                             int OH, int OW) {
    int b = blockIdx.z;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (h >= H || w >= W_in) return;

    for (int ic = 0; ic < IC; ++ic) {
        float val = 0.0f;
        for (int oc = 0; oc < OC; ++oc) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    int out_h = h - kh;
                    int out_w = w - kw;
                    if (out_h >= 0 && out_h < OH && out_w >= 0 && out_w < OW) {
                        int dout_idx = ((b * OH + out_h) * OW + out_w) * OC + oc;
                        int w_idx = ((oc * IC + ic) * KH + kh) * KW + kw;
                        val += d_out[dout_idx] * W[w_idx];
                    }
                }
            }
        }
        int idx = ((b * H + h) * W_in + w) * IC + ic;
        d_input[idx] = val;
    }
}


// ✅ Backward: dL/dKernel
__global__ void conv2d_backward_kernel_kernel(const float* input, const float* d_out,
                                              float* d_kernel, int B, int H, int W_in,
                                              int IC, int OC, int KH, int KW,
                                              int OH, int OW) {
    int oc = blockIdx.z;
    int ic = blockIdx.y;
    int kh = blockIdx.x;
    int kw = threadIdx.x;

    if (kh >= KH || kw >= KW) return;

    float sum = 0.0f;
    for (int b = 0; b < B; ++b) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                int in_h = oh + kh;
                int in_w = ow + kw;
                int in_idx = ((b * H + in_h) * W_in + in_w) * IC + ic;
                int out_idx = ((b * OH + oh) * OW + ow) * OC + oc;
                sum += input[in_idx] * d_out[out_idx];
            }
        }
    }

    int w_idx = ((oc * IC + ic) * KH + kh) * KW + kw;
    d_kernel[w_idx] = sum;
}

