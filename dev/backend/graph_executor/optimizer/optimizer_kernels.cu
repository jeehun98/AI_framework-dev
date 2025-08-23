// optimizer_kernels.cu
#include "optimizer_types.cuh"
#include <cuda_runtime.h>
#include <math.h>          // isfinite, sqrtf, powf, fminf/fmaxf
#include "../ge/logging_config.h"

#include "optimizer_config.cuh"
#include "optimizer_kernels.cuh"
#include "../ge/cuda_check.cuh"

// ===== 옵션 =====
#ifndef GRAD_CLIP_ENABLE          // 값 클리핑(절댓값 기준)
#define GRAD_CLIP_ENABLE 0
#endif
#ifndef GRAD_CLIP_THRESH
#define GRAD_CLIP_THRESH 1e4f
#endif

#ifndef GLOBAL_NORM_CLIP_ENABLE   // 글로벌 L2 노름 클리핑
#define GLOBAL_NORM_CLIP_ENABLE 0
#endif

#ifndef WEIGHT_DECAY_ENABLE       // Decoupled WD(AdamW/SGD-WD)
#define WEIGHT_DECAY_ENABLE 0
#endif

#ifndef NESTEROV_ENABLE           // 모멘텀의 Nesterov 모드
#define NESTEROV_ENABLE 0
#endif

#ifndef AMSGRAD_ENABLE            // Adam에서 vhat의 최대치 유지
#define AMSGRAD_ENABLE 0
#endif

#ifndef DEBUG_KERNEL
#define DEBUG_KERNEL 0
#endif

// ===== 공용 유틸 =====
__device__ __forceinline__ float value_clip(float g) {
#if GRAD_CLIP_ENABLE
    // 값 클리핑: 지나치게 큰 gradient를 [-T, +T]로 clamp
    const float T = GRAD_CLIP_THRESH;
    if (!isfinite(g)) return 0.0f;
    if (g >  T) return  T;
    if (g < -T) return -T;
#endif
    return g;
}

// ===== 글로벌 노름 클리핑(2-pass)용 리덕션 =====
#if GLOBAL_NORM_CLIP_ENABLE
__global__ void grad_sqsum_kernel(const float* __restrict__ grad, double* __restrict__ partial, int n) {
    extern __shared__ double s[];
    double sum = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float g = grad[i];
        if (isfinite(g)) {
            // 값 클리핑은 스케일 전에 하도록 동일 함수 사용
            g = value_clip(g);
            sum += (double)g * (double)g;
        }
    }
    s[threadIdx.x] = sum;
    __syncthreads();

    // block reduce
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = s[0];
}

__global__ void scale_grad_kernel(const float* __restrict__ grad_in, float* __restrict__ grad_out, float scale, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float g = value_clip(grad_in[i]);
        grad_out[i] = g * scale;
    }
}
#endif // GLOBAL_NORM_CLIP_ENABLE

// ===== SGD =====
__global__ void sgd_kernel(float* __restrict__ param,
                           const float* __restrict__ grad,
#if WEIGHT_DECAY_ENABLE
                           float weight_decay,
#endif
                           float lr, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float g = value_clip(grad[i]);
        if (!isfinite(g)) continue;

#if WEIGHT_DECAY_ENABLE
        // Decoupled WD: p = p - lr*(g) - lr*wd*p
        float p = param[i];
        p -= lr * g;
        p -= lr * weight_decay * p;
        param[i] = p;
#else
        param[i] -= lr * g;
#endif

#if DEBUG_KERNEL
        if (i == 0) KPRINTF("[SGD] lr=%g, g0=%g, p0=%g\n", lr, g, param[i]);
#endif
    }
}

// ===== Momentum (classic/Nesterov) =====
__global__ void momentum_kernel(float* __restrict__ param,
                                const float* __restrict__ grad,
                                float* __restrict__ velocity,
#if WEIGHT_DECAY_ENABLE
                                float weight_decay,
#endif
                                float lr, float beta, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float g = value_clip(grad[i]);
        if (!isfinite(g)) continue;

#if WEIGHT_DECAY_ENABLE
        // decoupled WD
        float p = param[i];
        p -= lr * weight_decay * p;
        param[i] = p;
#endif

        float v = beta * velocity[i] + g;
        velocity[i] = v;

#if NESTEROV_ENABLE
        float upd = lr * (beta * v + g);  // Nesterov
#else
        float upd = lr * v;               // classic
#endif
        if (!isfinite(upd)) continue;
        param[i] -= upd;

#if DEBUG_KERNEL
        if (i == 0) KPRINTF("[MOMENTUM] lr=%g beta=%g | g0=%g v0=%g p0=%g\n", lr, beta, g, v, param[i]);
#endif
    }
}

// ===== Adam / AdamW (+ AMSGrad) =====
__global__ void adam_kernel(float* __restrict__ param,
                            const float* __restrict__ grad,
                            float* __restrict__ m,
                            float* __restrict__ v,
#if AMSGRAD_ENABLE
                            float* __restrict__ vhat_max,
#endif
#if WEIGHT_DECAY_ENABLE
                            float weight_decay,
#endif
                            float lr, float beta1, float beta2, float eps,
                            int t, int n) {
    const int t_eff = (t < 1) ? 1 : t;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float g = value_clip(grad[i]);
        if (!isfinite(g)) continue;

        float mi = beta1 * m[i] + (1.0f - beta1) * g;
        float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
        m[i] = mi;
        v[i] = vi;

        // bias correction
        float m_hat = mi / fmaxf(1.0f - powf(beta1, (float)t_eff), 1e-12f);
        float v_hat = vi / fmaxf(1.0f - powf(beta2, (float)t_eff), 1e-12f);

#if AMSGRAD_ENABLE
        float vmax = fmaxf(v_hat, vhat_max[i]);
        vhat_max[i] = vmax;
        v_hat = vmax;
#endif
        float denom = sqrtf(fmaxf(v_hat, 1e-12f)) + eps;
        float step  = lr * (m_hat / denom);
        if (!isfinite(step)) continue;

#if WEIGHT_DECAY_ENABLE
        // AdamW: decoupled WD
        float p = param[i];
        p -= lr * weight_decay * p;
        p -= step;
        param[i] = p;
#else
        param[i] -= step;
#endif

#if DEBUG_KERNEL
        if (i == 0 && (t_eff % 100 == 0)) {
            KPRINTF("[ADAM] t=%d lr=%g b1=%g b2=%g eps=%g | g0=%g m=%g v=%g m^=%g v^=%g step=%g p0=%g\n",
                t_eff, lr, beta1, beta2, eps, g, mi, vi, m_hat, v_hat, step, param[i]);
        }
#endif
    }
}

void optimizer_update_cuda(
    float* param,
    const float* grad,
    float* velocity,
    float* m,
    float* v,
#if AMSGRAD_ENABLE
    float* vhat_max,
#endif
    float lr, float beta1, float beta2, float eps,
#if WEIGHT_DECAY_ENABLE
    float weight_decay,
#endif
    int size,
    OptimizerType opt_type,
    int timestep,
    cudaStream_t stream)
{
    const int threads = 256;
    const int blocks  = (size + threads - 1) / threads;

#if GLOBAL_NORM_CLIP_ENABLE
    // ---- 안전한 호출-로컬 워크스페이스 사용 ----
    float*  grad_scaled = nullptr;
    CUDA_CHECK(cudaMalloc(&grad_scaled, size * sizeof(float)));

    // partial sums (device)
    int redBlocks = min(blocks, 1024);
    double* d_partial = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partial, redBlocks * sizeof(double)));

    // 1) ∥g∥₂^2 부분합
    size_t shmem = threads * sizeof(double);
    grad_sqsum_kernel<<<redBlocks, threads, shmem, stream>>>(grad, d_partial, size);
    CUDA_CHECK(cudaGetLastError());

    // 2) host reduce (async copy + sync)
    std::vector<double> h_partial(redBlocks);
    CUDA_CHECK(cudaMemcpyAsync(h_partial.data(), d_partial,
                               redBlocks * sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream)); // ← 반드시 동기화
    double sum = 0.0;
    for (int i = 0; i < redBlocks; ++i) sum += h_partial[i];
    CUDA_CHECK(cudaFree(d_partial));

    const double norm = sqrt(sum + 1e-30);
    float scale = 1.0f;
    const float clip_threshold = GRAD_CLIP_THRESH;
    if (norm > (double)clip_threshold) {
        scale = (float)((double)clip_threshold / norm);
    }

    // 3) g_scaled = scale * clip(g)
    scale_grad_kernel<<<blocks, threads, 0, stream>>>(grad, grad_scaled, scale, size);
    CUDA_CHECK(cudaGetLastError());
    const float* gptr = grad_scaled;
#else
    const float* gptr = grad;
#endif

    switch (opt_type) {
        case OptimizerType::SGD:
            sgd_kernel<<<blocks, threads, 0, stream>>>(param, gptr,
#if WEIGHT_DECAY_ENABLE
                weight_decay,
#endif
                lr, size);
            CUDA_CHECK(cudaGetLastError());
            break;

        case OptimizerType::MOMENTUM:
            if (!velocity) { 
#if GLOBAL_NORM_CLIP_ENABLE
                CUDA_CHECK(cudaFree((void*)gptr));
#endif
                return;
            }
            momentum_kernel<<<blocks, threads, 0, stream>>>(param, gptr, velocity,
#if WEIGHT_DECAY_ENABLE
                weight_decay,
#endif
                lr, beta1 /*momentum coeff*/, size);
            CUDA_CHECK(cudaGetLastError());
            break;

        case OptimizerType::ADAM:
            if (!m || !v) {
#if GLOBAL_NORM_CLIP_ENABLE
                CUDA_CHECK(cudaFree((void*)gptr));
#endif
                return;
            }
            adam_kernel<<<blocks, threads, 0, stream>>>(param, gptr, m, v,
#if AMSGRAD_ENABLE
                vhat_max,
#endif
#if WEIGHT_DECAY_ENABLE
                weight_decay,
#endif
                lr, beta1, beta2, eps, timestep, size);
            CUDA_CHECK(cudaGetLastError());
            break;

        default:
#if GLOBAL_NORM_CLIP_ENABLE
            CUDA_CHECK(cudaFree((void*)gptr));
#endif
            return;
    }

#if GLOBAL_NORM_CLIP_ENABLE
    CUDA_CHECK(cudaFree((void*)gptr)); // grad_scaled 해제
#endif

#if DEBUG_KERNEL
    CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
}
