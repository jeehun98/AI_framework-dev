// pooling_ops.cu
#include "pooling_ops.cuh"
#include <cuda_runtime.h>

static inline dim3 make_grid_1d(int total, int block = 256) {
    return dim3((total + block - 1) / block);
}

// Forward MaxPool
void maxpool2d_forward(
    const float* x, float* y, int32_t* argmax,
    const Pool2DParams& p, cudaStream_t stream)
{
    const int total = p.N * p.C * p.H_out * p.W_out;
    const int block = 256;
    maxpool2d_forward_kernel<<<make_grid_1d(total, block), block, 0, stream>>>(x, y, argmax, p);
}

// Forward AvgPool
void avgpool2d_forward(
    const float* x, float* y,
    const Pool2DParams& p, cudaStream_t stream)
{
    const int total = p.N * p.C * p.H_out * p.W_out;
    const int block = 256;
    avgpool2d_forward_kernel<<<make_grid_1d(total, block), block, 0, stream>>>(x, y, p);
}

// Backward MaxPool
void maxpool2d_backward(
    const float* grad_y, float* grad_x, const int32_t* argmax,
    const Pool2DParams& p, cudaStream_t stream)
{
    const int total = p.N * p.C * p.H_out * p.W_out;
    const int block = 256;
    maxpool2d_backward_kernel<<<make_grid_1d(total, block), block, 0, stream>>>(grad_y, grad_x, argmax, p);
}

// Backward AvgPool
void avgpool2d_backward(
    const float* grad_y, float* grad_x,
    const Pool2DParams& p, cudaStream_t stream)
{
    const int total = p.N * p.C * p.H_out * p.W_out;
    const int block = 256;
    avgpool2d_backward_kernel<<<make_grid_1d(total, block), block, 0, stream>>>(grad_y, grad_x, p);
}
