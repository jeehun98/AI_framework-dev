#ifndef BACKWARD_KERNELS_CUH
#define BACKWARD_KERNELS_CUH

// 기존 커널
__global__ void matmul_backward_input(const float* d_out, const float* W_T, float* d_input, int M, int N, int K);
__global__ void matmul_backward_weight(const float* input_T, const float* d_out, float* d_weight, int M, int N, int K);
__global__ void add_backward_bias(const float* d_out, float* d_bias, int rows, int cols);
__global__ void add_backward_input(const float* d_out, float* d_input, int size);
__global__ void fill_gradient(float* grad, int total_size, float value);

// ✅ 여기에 선언만
__global__ void fill_gradient(float* d_grad, int size, float value);

#endif
