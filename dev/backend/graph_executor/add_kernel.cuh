#ifndef ADD_KERNEL_CUH
#define ADD_KERNEL_CUH

#include <cuda_runtime.h>

// CUDA 커널 함수 선언
__global__ void add_kernel(const float* input, const float* bias, float* output, int rows, int cols);

#endif // ADD_KERNEL_CUH
