// kernels/epilogue_kernel.cu
#include <cuda_runtime.h>
#include "epilogue_params.cuh"

__device__ inline float relu(float v){ return v > 0.f ? v : 0.f; }

// -------------------------
// [A] Generic (기존 유지)
// -------------------------
extern "C" __global__
void epilogue_kernel_f32_rowmajor(EpParams P){
  int M = P.M, N = P.N;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * N;
  for (int i = idx; i < total; i += gridDim.x * blockDim.x) {
    int m = i / N, n = i % N;
    int ix = m * P.ld_x + n;
    int iy = m * P.ld_y + n;
    float v = P.x[ix];
    if (P.has_bias) v += P.bias[n];
    if (P.act == 1) v = relu(v);   // ReLU
    // (다른 Act는 추후 추가)
    float out = P.alpha * v + (P.beta != 0.f ? P.beta * P.y[iy] : 0.f);
    P.y[iy] = out;
  }
}

// ---------------------------------------------
// [B] Optimized, compile-time specialized paths
//    - ReLU + Bias(PerN)
//    - ReLU + NoBias
// ---------------------------------------------

// ReLU + Bias(PerN)
extern "C" __global__
void epilogue_kernel_f32_rowmajor_relu_bias(EpParams P){
  int M = P.M, N = P.N;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * N;
  for (int i = idx; i < total; i += gridDim.x * blockDim.x) {
    int m = i / N, n = i % N;
    int ix = m * P.ld_x + n;
    int iy = m * P.ld_y + n;
    float v = P.x[ix] + P.bias[n];   // bias 고정
    v = v > 0.f ? v : 0.f;           // ReLU 고정
    float out = P.alpha * v + (P.beta != 0.f ? P.beta * P.y[iy] : 0.f);
    P.y[iy] = out;
  }
}

// ReLU + NoBias
extern "C" __global__
void epilogue_kernel_f32_rowmajor_relu_nobias(EpParams P){
  int M = P.M, N = P.N;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * N;
  for (int i = idx; i < total; i += gridDim.x * blockDim.x) {
    int m = i / N, n = i % N;
    int ix = m * P.ld_x + n;
    int iy = m * P.ld_y + n;
    float v = P.x[ix];               // bias 없음
    v = v > 0.f ? v : 0.f;           // ReLU 고정
    float out = P.alpha * v + (P.beta != 0.f ? P.beta * P.y[iy] : 0.f);
    P.y[iy] = out;
  }
}
