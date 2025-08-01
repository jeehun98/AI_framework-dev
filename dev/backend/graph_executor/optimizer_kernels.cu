#include "optimizer_types.cuh"
#include <cuda_runtime.h>

__global__ void sgd_kernel(float* param, float* grad, float lr, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        param[i] -= lr * grad[i];
    }
}

__global__ void momentum_kernel(float* param, float* grad, float* velocity, float lr, float beta, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) return;

    velocity[i] = beta * velocity[i] + (1.0f - beta) * grad[i];
    param[i] -= lr * velocity[i];
}

__global__ void adam_kernel(float* param, float* grad, float* m, float* v,
                            float lr, float beta1, float beta2, float epsilon, int t, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
    v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];

    float m_hat = m[i] / (1.0f - powf(beta1, t));
    float v_hat = v[i] / (1.0f - powf(beta2, t));

    param[i] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
}

void optimizer_update_cuda(
    float* param, float* grad,
    float* velocity, float* m, float* v,
    float lr, float beta1, float beta2, float epsilon,
    int size, OptimizerType opt_type, int t
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    switch (opt_type) {
        case SGD:
            sgd_kernel<<<blocks, threads>>>(param, grad, lr, size);
            break;
        case MOMENTUM:
            if (velocity == nullptr) return;
            momentum_kernel<<<blocks, threads>>>(param, grad, velocity, lr, beta1, size);
            break;
        case ADAM:
            if (m == nullptr || v == nullptr) return;
            adam_kernel<<<blocks, threads>>>(param, grad, m, v, lr, beta1, beta2, epsilon, t, size);
            break;
        default:
            break;
    }

    cudaDeviceSynchronize();
}
