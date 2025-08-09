// optimizer_kernels.cu

#include "optimizer_types.cuh"
#include <cuda_runtime.h>
#include <math.h>      // isnan, isinf, sqrtf, powf
#include "logging_config.h"   // KPRINTF/LOGV

// ---- 옵션: 그라디언트 클리핑 토글/임계값 ----
#ifndef GRAD_CLIP_ENABLE
#define GRAD_CLIP_ENABLE 0   // 0: off, 1: on
#endif
#ifndef GRAD_CLIP_THRESH
#define GRAD_CLIP_THRESH 1e4f
#endif

// ---- 공용: 안전 체크(필요 시만 로그) ----
__device__ inline float clip_grad(float g) {
#if GRAD_CLIP_ENABLE
    if (!isfinite(g) || fabsf(g) > GRAD_CLIP_THRESH) {
        // 너무 잦은 로그 방지: 앞 몇 개만
        // KPRINTF("[opt][clip] g=%e -> 0\n", g);
        return 0.0f;
    }
#endif
    return g;
}

// -------------------- SGD --------------------
__global__ void sgd_kernel(float* __restrict__ param,
                           const float* __restrict__ grad,
                           float lr, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    float g = grad[i];
    g = clip_grad(g);

    if (!isfinite(g)) return;

    param[i] -= lr * g;

#if DEBUG_KERNEL
    if (i == 0) {
        KPRINTF("[SGD] lr=%g, g0=%g, p0=%g\n", lr, g, param[i]);
    }
#endif
}

// --------------- Momentum (classic) ---------------
// 필요에 따라 EMA 스타일이 아니라 고전식 v = beta*v + grad 로 구현
__global__ void momentum_kernel(float* __restrict__ param,
                                const float* __restrict__ grad,
                                float* __restrict__ velocity,
                                float lr, float beta, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    float g = grad[i];
    g = clip_grad(g);
    if (!isfinite(g)) return;

    velocity[i] = beta * velocity[i] + g;  // 고전식
    float upd = lr * velocity[i];

    if (!isfinite(upd)) return;

    param[i] -= upd;

#if DEBUG_KERNEL
    if (i == 0) {
        KPRINTF("[MOMENTUM] lr=%g beta=%g | g0=%g v0=%g p0=%g\n", lr, beta, g, velocity[i], param[i]);
    }
#endif
}

// -------------------- Adam --------------------
__global__ void adam_kernel(float* __restrict__ param,
                            const float* __restrict__ grad,
                            float* __restrict__ m,
                            float* __restrict__ v,
                            float lr, float beta1, float beta2, float epsilon,
                            int t, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    // t는 편향보정에 1부터 들어가야 안정
    int t_eff = (t < 1) ? 1 : t;

    float g = grad[i];
    g = clip_grad(g);
    if (!isfinite(g)) return;

    // 1차/2차 모멘트
    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;

    m[i] = mi;
    v[i] = vi;

    // 편향 보정
    float m_hat_denom = 1.0f - powf(beta1, (float)t_eff);
    float v_hat_denom = 1.0f - powf(beta2, (float)t_eff);
    // 극단 보호
    m_hat_denom = fmaxf(m_hat_denom, 1e-8f);
    v_hat_denom = fmaxf(v_hat_denom, 1e-8f);

    float m_hat = mi / m_hat_denom;
    float v_hat = vi / v_hat_denom;

    // 업데이트
    float denom = sqrtf(fmaxf(v_hat, 1e-12f)) + epsilon;
    if (!isfinite(denom)) return;

    float update = lr * (m_hat / denom);
    if (!isfinite(update)) return;

    param[i] -= update;

#if DEBUG_KERNEL
    if (i == 0 && (t_eff % 100 == 0)) {
        KPRINTF("[ADAM] t=%d lr=%g b1=%g b2=%g eps=%g | g0=%g m=%g v=%g m^=%g v^=%g upd=%g p0=%g\n",
                t_eff, lr, beta1, beta2, epsilon, g, mi, vi, m_hat, v_hat, update, param[i]);
    }
#endif
}

// -------------------- Host Launcher --------------------
void optimizer_update_cuda(
    float* param,
    const float* grad,   // ★ const
    float* velocity,
    float* m,
    float* v,
    float lr, float beta1, float beta2, float epsilon,
    int size, OptimizerType opt_type, int t) 
{
    int threads = 256;
    int blocks  = (size + threads - 1) / threads;

    switch (opt_type) {
        case SGD:
            sgd_kernel<<<blocks, threads>>>(param, grad, lr, size);
            break;

        case MOMENTUM:
            if (velocity == nullptr) return;
            momentum_kernel<<<blocks, threads>>>(param, grad, velocity, lr, beta1 /*as beta*/, size);
            break;

        case ADAM:
            if (m == nullptr || v == nullptr) return;
            adam_kernel<<<blocks, threads>>>(param, grad, m, v, lr, beta1, beta2, epsilon, t, size);
            break;

        default:
            return;
    }

    // 디버깅 레벨에 따라 동기화 제어 (기본 조용)
#if DEBUG_KERNEL
    cudaDeviceSynchronize();
#endif
}
