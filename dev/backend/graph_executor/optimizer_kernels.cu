// optimizer_kernels.cu

#include "optimizer_types.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>  // for isnan, sqrtf, etc.

__global__ void sgd_kernel(float* param, float* grad, float lr, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        param[i] -= lr * grad[i];
    }
}

__global__ void momentum_kernel(float* param, float* grad, float* velocity, float lr, float beta, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    velocity[i] = beta * velocity[i] + (1.0f - beta) * grad[i];
    param[i] -= lr * velocity[i];
}

__global__ void adam_kernel(float* param, float* grad, float* m, float* v,
                            float lr, float beta1, float beta2, float epsilon, int t, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    float g = grad[i];
    if (isnan(g) || isinf(g)) {
        printf("[adam] grad[%d] = %f (NaN or Inf)\n", i, g);
        return;
    }

    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

    float m_hat_denom = fmaxf(1.0f - powf(beta1, t), 1e-8f);
    float v_hat_denom = fmaxf(1.0f - powf(beta2, t), 1e-8f);

    float m_hat = m[i] / m_hat_denom;
    float v_hat = v[i] / v_hat_denom;

    float denom = sqrtf(v_hat) + epsilon;
    if (isnan(denom) || denom < 1e-8f) {
        printf("[adam] sqrt(v_hat)+eps unstable → denom=%f, v_hat=%f, epsilon=%f\n", denom, v_hat, epsilon);
        return;
    }

    float update = lr * m_hat / denom;
    if (isnan(update) || isinf(update)) {
        printf("[adam] update[%d] = %f (NaN or Inf)\n", i, update);
        return;
    }

    param[i] -= update;

    if (isnan(param[i]) || isinf(param[i])) {
        printf("[adam] param[%d] became NaN or Inf after update=%f\n", i, update);
    }

    if (i == 0 && t % 100 == 0) {
        printf("[adam][t=%d i=%d] grad=%f m=%f v=%f m?=%f v?=%f update=%f param=%f\n",
               t, i, g, m[i], v[i], m_hat, v_hat, update, param[i]);
    }
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
