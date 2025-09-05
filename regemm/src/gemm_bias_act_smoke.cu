#include <cuda_runtime.h>
#include "regemm/api.h"
#include "regemm/activations.h"
#include "regemm/bias.h"

namespace regemm {

static __device__ __forceinline__ float apply_act(float x, ActKind k){
  if (k == ActKind::ReLU) return act_relu(x);
  return act_none(x);
}

__global__ void gemm_bias_act_f32_smoke(GemmBiasActParams p) {
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= p.M || n >= p.N) return;

  const float* A = reinterpret_cast<const float*>(p.A);
  const float* B = reinterpret_cast<const float*>(p.B);
  const float* C = reinterpret_cast<const float*>(p.C);
  float*       D = reinterpret_cast<float*>(p.D);

  // 1) 누산 (in-register)
  float acc = 0.f;
  #pragma unroll 1
  for (int k = 0; k < p.K; ++k) {
    float a = A[m * p.lda + k];
    float b = B[k * p.ldb + n];
    acc = fmaf(a, b, acc); // acc += a*b
  }

  // 2) alpha/beta, 3) C_in (옵션) in-register
  acc *= p.alpha;
  if (p.beta != 0.f && C) {
    float cin = C[m * p.ldc + n];
    acc = fmaf(p.beta, cin, acc); // acc += beta*Cin
  }

  // 4) bias in-register
  acc += load_bias(p, m, n);

  // 5) activation in-register
  acc = apply_act(acc, p.act);

  // 6) 최초/유일 global store
  D[m * p.ldd + n] = acc;
}

void launch_gemm_bias_act_f32_smoke(const GemmBiasActParams& p, cudaStream_t s){
  dim3 block(16, 16);
  dim3 grid((p.N + block.x - 1)/block.x, (p.M + block.y - 1)/block.y);
  gemm_bias_act_f32_smoke<<<grid, block, 0, s>>>(p);
}

} // namespace regemm
