#include <cuda_runtime.h>
#include <math.h>
#include "activation_ops.cuh"

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

// ---- 공통 유틸 -------------------------------------------------------------

__device__ __forceinline__ float sigmoid_stable(float x) {
    if (!isfinite(x)) return 0.5f;
    // 오버/언더플로우를 피하는 안전한 구현
    float z = __expf(-fabsf(x));
    float s = (x >= 0.f) ? 1.f/(1.f+z) : z/(1.f+z);
    const float eps = 1e-7f;
    return fminf(fmaxf(s, eps), 1.f - eps);
}

__device__ __forceinline__ float act_forward(int act_type, float z) {
    switch (act_type) {
        case ACT_RELU:    return fmaxf(z, 0.f);
        case ACT_SIGMOID: return sigmoid_stable(z);
        case ACT_TANH: {
            float t = tanhf(z);
            return isfinite(t) ? t : 0.f;
        }
        default:          return z; // identity 폴백
    }
}

__device__ __forceinline__ float act_backward_from_out(int act_type, float out, float go) {
    // out == f(z) (forward 결과) 를 이용해 미분
    switch (act_type) {
        case ACT_RELU:    return (out > 0.f) ? go : 0.f;           // z>0 ↔ out>0
        case ACT_SIGMOID: return go * out * (1.f - out);           // σ'(z)=o(1-o)
        case ACT_TANH:    return go * (1.f - out * out);           // 1 - tanh^2
        default:          return 0.f;
    }
}

// ---- Forward kernel --------------------------------------------------------

__global__ void activation_forward_kernel(const float* __restrict__ in,
                                          const float* __restrict__ bias, // [cols] or null
                                          float* __restrict__ out,
                                          int rows, int cols, int act_type)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= cols) return;

    int idx = row * cols + col;
    float z = in[idx] + (bias ? bias[col] : 0.f);
    out[idx] = act_forward(act_type, z);
}

// ---- Backward kernel -------------------------------------------------------

__global__ void activation_backward_kernel(const float* __restrict__ grad_out,
                                           const float* __restrict__ out,   // forward output
                                           float* __restrict__ grad_in,
                                           int rows, int cols, int act_type)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= cols) return;

    int idx = row * cols + col;
    float go = grad_out[idx];
    float o  = out[idx];

    if (!isfinite(go) || !isfinite(o)) {
        if (idx == 0) KPRINTF("[act_bw][NaN/Inf] go=%.6f o=%.6f act=%d\n", go, o, act_type);
        grad_in[idx] = 0.f;
        return;
    }

    float gi = act_backward_from_out(act_type, o, go);

    if (!isfinite(gi) || fabsf(gi) > 1e10f) {
        if (idx == 0) KPRINTF("[act_bw] clamp gi=%.6f (go=%.6f o=%.6f)\n", gi, go, o);
        gi = 0.f;
    }
    grad_in[idx] = gi;
}

// ---- Launchers -------------------------------------------------------------

void launch_activation_forward(const float* in, const float* bias, float* out,
                               int rows, int cols, int act_type,
                               cudaStream_t stream)
{
    dim3 block(ACT_BLOCK_X, ACT_BLOCK_Y);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);
    activation_forward_kernel<<<grid, block, 0, stream>>>(in, bias, out, rows, cols, act_type);
}

void launch_activation_backward(const float* grad_out, const float* out, float* grad_in,
                                int rows, int cols, int act_type,
                                cudaStream_t stream)
{
    dim3 block(ACT_BLOCK_X, ACT_BLOCK_Y);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);
    activation_backward_kernel<<<grid, block, 0, stream>>>(grad_out, out, grad_in, rows, cols, act_type);
}
