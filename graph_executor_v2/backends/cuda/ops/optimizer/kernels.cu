// backends/cuda/ops/optimizer/kernels.cu
#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>   // powf, sqrtf

// ----------------------------------------------
// 내부 유틸 (rmsnorm/kernels.cu와 유사 톤)
// ----------------------------------------------
namespace {

static inline __device__ int gtid() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

static inline dim3 pick_block(int64_t /*N*/) {
  return dim3(256);
}

static inline dim3 pick_grid(int64_t N) {
  int64_t blocks = (N + 256 - 1) / 256;
  if (blocks < 32)   blocks = 32;     // 캡처 안전: SM 조회 없이 보수적 그리드
  if (blocks > 4096) blocks = 4096;   // 안전 상한
  return dim3(static_cast<unsigned>(blocks));
}

// ============================================================
// SGD + Momentum (+Nesterov, L2 weight decay)
//   P <- P - lr * g' (momentum/nesterov 반영)
//   V <- mu * V + (1-damp) * g
//   g  <- g + wd * P  (L2)
//   if nesterov: step_grad = g + mu * V; else step_grad = V
// ============================================================
__global__ void sgd_update_kernel(
    float* __restrict__ P,          // [N]
    const float* __restrict__ G,    // [N]
    float* __restrict__ V,          // [N] or nullptr
    int64_t N,
    float lr, float momentum, float dampening,
    int nesterov, float weight_decay)
{
  for (int i = gtid(); i < N; i += blockDim.x * gridDim.x) {
    float p = P[i];
    float g = G[i];

    // L2 decay
    if (weight_decay != 0.0f) {
      g += weight_decay * p;
    }

    if (momentum > 0.0f && V != nullptr) {
      float v = momentum * V[i] + (1.0f - dampening) * g;
      V[i] = v;
      float step_grad = nesterov ? (g + momentum * v) : v;
      p -= lr * step_grad;
    } else {
      p -= lr * g;
    }

    P[i] = p;
  }
}

// ============================================================
// AdamW (Decoupled Weight Decay)
//   p <- p - lr * wd * p
//   m <- b1*m + (1-b1)*g
//   v <- b2*v + (1-b2)*g^2
//   if bias_correction:
//       mhat = m / (1 - b1^t)
//       vhat = v / (1 - b2^t)
//   p <- p - lr * mhat / (sqrt(vhat) + eps)
// ============================================================
__global__ void adamw_update_kernel(
    float* __restrict__ P,
    const float* __restrict__ G,
    float* __restrict__ M,
    float* __restrict__ Vt,
    int64_t N,
    float lr, float beta1, float beta2, float eps,
    float weight_decay, int bias_correction, int step)
{
  // bias correction 계수 (모든 스레드 동일 값)
  float inv_b1t = 1.0f;
  float inv_b2t = 1.0f;
  if (bias_correction) {
    // step >= 1 가정
    float b1t = 1.0f - powf(beta1, (float)step);
    float b2t = 1.0f - powf(beta2, (float)step);
    // 안전상 0 division 방지(극단적 베타/스텝)
    if (b1t < 1e-12f) b1t = 1e-12f;
    if (b2t < 1e-12f) b2t = 1e-12f;
    inv_b1t = 1.0f / b1t;
    inv_b2t = 1.0f / b2t;
  }

  for (int i = gtid(); i < N; i += blockDim.x * gridDim.x) {
    float p = P[i];
    float g = G[i];

    // decoupled weight decay
    if (weight_decay != 0.0f) {
      p -= lr * weight_decay * p;
    }

    float m = M[i] = beta1 * M[i] + (1.0f - beta1) * g;
    float v = Vt[i] = beta2 * Vt[i] + (1.0f - beta2) * (g * g);

    float mhat = bias_correction ? (m * inv_b1t) : m;
    float vhat = bias_correction ? (v * inv_b2t) : v;

    p -= lr * (mhat / (sqrtf(vhat) + eps));
    P[i] = p;
  }
}

} // anonymous namespace

// ----------------------------------------------
// RMSNorm과 동일한 네임스페이스/패턴: ai::launcher()
// ----------------------------------------------
namespace ai {

void sgd_update_kernel_launcher(
    float* P,
    const float* G,
    float* V,
    int64_t N,
    float lr, float momentum, float dampening,
    int nesterov, float weight_decay,
    cudaStream_t s)
{
  if (N <= 0) return;
  dim3 block = pick_block(N);
  dim3 grid  = pick_grid(N);
  sgd_update_kernel<<<grid, block, 0, s>>>(
      P, G, V, N, lr, momentum, dampening, nesterov, weight_decay);
}

void adamw_update_kernel_launcher(
    float* P, const float* G, float* M, float* V,
    int64_t N,
    float lr, float beta1, float beta2, float eps,
    float weight_decay, int bias_correction, int step,
    cudaStream_t s)
{
  if (N <= 0) return;
  dim3 block = pick_block(N);
  dim3 grid  = pick_grid(N);
  adamw_update_kernel<<<grid, block, 0, s>>>(
      P, G, M, V, N, lr, beta1, beta2, eps, weight_decay, bias_correction, step);
}

} // namespace ai
