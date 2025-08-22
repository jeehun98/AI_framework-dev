// activation_ops.cu
#include "activation_ops.cuh"

#include <cuda_runtime.h>
#include <math.h>     // fabsf, tanhf, expf, sqrtf, isfinite

// 선택: 로깅 매크로가 있으면 사용
#ifdef __has_include
#  if __has_include("logging_config.h")
#    include "logging_config.h"
#  else
#    define KPRINTF(...) ((void)0)
#  endif
#else
#  define KPRINTF(...) ((void)0)
#endif

// ----------------------- Device utilities -----------------------------------

__device__ __forceinline__ float sigmoid_stable(float x) {
    if (!isfinite(x)) return 0.5f;
    // 오버/언더플로우 방지: σ(x)=1/(1+e^{-x})
    float z = __expf(-fabsf(x));
    float s = (x >= 0.f) ? 1.f / (1.f + z) : z / (1.f + z);
    const float eps = 1e-7f;
    return fminf(fmaxf(s, eps), 1.f - eps);
}

// GELU (tanh 근사)
// 0.5 * x * (1 + tanh( √(2/π) * (x + 0.044715 x^3) ))
__device__ __forceinline__ float gelu_tanh_forward(float x) {
    const float k0 = 0.7978845608028654f;  // sqrt(2/pi)
    const float k1 = 0.044715f;
    float x3 = x * x * x;
    float u  = k0 * (x + k1 * x3);
    return 0.5f * x * (1.f + tanhf(u));
}

__device__ __forceinline__ float gelu_tanh_backward(float x, float go) {
    const float k0 = 0.7978845608028654f;
    const float k1 = 0.044715f;
    float x2 = x * x;
    float x3 = x2 * x;
    float u  = k0 * (x + k1 * x3);
    float t  = tanhf(u);
    float sech2 = 1.f - t * t;          // sech^2(u)
    float du_dx = k0 * (1.f + 3.f * k1 * x2);
    float dgelu_dx = 0.5f * (1.f + t) + 0.5f * x * sech2 * du_dx;
    return go * dgelu_dx;
}

__device__ __forceinline__ float act_forward_val(int act_type, float z, float alpha, int gelu_tanh_flag) {
    switch (act_type) {
        case ACT_RELU:    return fmaxf(z, 0.f);
        case ACT_SIGMOID: return sigmoid_stable(z);
        case ACT_TANH: {
            float t = tanhf(z);
            return isfinite(t) ? t : 0.f;
        }
        case ACT_LEAKY:   return (z >= 0.f) ? z : alpha * z;
        case ACT_ELU:     return (z >= 0.f) ? z : alpha * (expf(z) - 1.f);
        case ACT_GELU:    // 여기서는 tanh 근사만 사용
            return gelu_tanh_forward(z);
        case ACT_SILU:    return z * sigmoid_stable(z);  // Swish
        default:          return z; // identity
    }
}

// out=f(z)와 z를 모두 사용할 수 있게 설계(ELU/GELU 안정성↑)
__device__ __forceinline__ float act_backward_from_out(int act_type, float z, float out, float go, float alpha, int gelu_tanh_flag) {
    switch (act_type) {
        case ACT_RELU:    return (z > 0.f) ? go : 0.f;
        case ACT_SIGMOID: return go * out * (1.f - out);
        case ACT_TANH:    return go * (1.f - out * out);
        case ACT_LEAKY:   return go * ((z >= 0.f) ? 1.f : alpha);
        case ACT_ELU:     // out = alpha*(exp(z)-1) (z<0), d/dz = out + alpha
            return go * ((z >= 0.f) ? 1.f : (out + alpha));
        case ACT_GELU:    return gelu_tanh_backward(z, go);
        case ACT_SILU: {  // y = x*s, s=sigmoid(x) → dy/dx = s + x*s*(1-s)
            float s = sigmoid_stable(z);
            return go * (s + z * s * (1.f - s));
        }
        default:          return go; // identity
    }
}

// ---------------------------- Kernels ---------------------------------------

__global__ void activation_forward_kernel(const float* __restrict__ in,
                                          const float* __restrict__ bias, // [cols] or null
                                          float* __restrict__ out,
                                          int rows, int cols,
                                          int act_type, float alpha, int gelu_tanh_flag)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= cols) return;

    int idx = row * cols + col;
    float z = in[idx] + (bias ? bias[col] : 0.f);
    float y = act_forward_val(act_type, z, alpha, gelu_tanh_flag);

    // 간단한 NaN/Inf 가드
    if (!isfinite(y)) {
        if (idx == 0) KPRINTF("[act_fw][NaN/Inf] z=%.6f act=%d\n", z, act_type);
        y = 0.f;
    }
    out[idx] = y;
}

__global__ void activation_backward_kernel(const float* __restrict__ grad_out,
                                           const float* __restrict__ in,   // pre-activation z
                                           const float* __restrict__ out,  // f(z)
                                           float* __restrict__ grad_in,
                                           int rows, int cols,
                                           int act_type, float alpha, int gelu_tanh_flag)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= cols) return;

    int idx = row * cols + col;
    float go = grad_out[idx];
    float z  = in[idx];
    float o  = out[idx];

    if (!isfinite(go) || !isfinite(z) || !isfinite(o)) {
        if (idx == 0) KPRINTF("[act_bw][NaN/Inf] go=%.6f z=%.6f o=%.6f act=%d\n", go, z, o, act_type);
        grad_in[idx] = 0.f;
        return;
    }

    float gi = act_backward_from_out(act_type, z, o, go, alpha, gelu_tanh_flag);

    // 폭주/NaN 방지
    if (!isfinite(gi) || fabsf(gi) > 1e10f) {
        if (idx == 0) KPRINTF("[act_bw] clamp gi=%.6f (go=%.6f z=%.6f o=%.6f)\n", gi, go, z, o);
        gi = 0.f;
    }
    grad_in[idx] = gi;
}

// ---------------------------- Launchers -------------------------------------

void launch_activation_forward(const float* in, const float* bias, float* out,
                               int rows, int cols, int act_type,
                               float alpha, int gelu_tanh_flag,
                               cudaStream_t stream)
{
    dim3 block(ACT_BLOCK_X, ACT_BLOCK_Y);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);
    activation_forward_kernel<<<grid, block, 0, stream>>>(
        in, bias, out, rows, cols, act_type, alpha, gelu_tanh_flag
    );
}

void launch_activation_backward(const float* grad_out,
                                const float* in,      // pre-activation z
                                const float* out,     // f(z)
                                float* grad_in,
                                int rows, int cols, int act_type,
                                float alpha, int gelu_tanh_flag,
                                cudaStream_t stream)
{
    dim3 block(ACT_BLOCK_X, ACT_BLOCK_Y);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);
    activation_backward_kernel<<<grid, block, 0, stream>>>(
        grad_out, in, out, grad_in, rows, cols, act_type, alpha, gelu_tanh_flag
    );
}
