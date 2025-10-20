#include <cuda_runtime.h>
#include "policy/ep_functors.cuh"
#include "policy/ep_kernel_policy.cuh"
#include "policy/ep_policy.cuh"
#include "policy/ep_traits.cuh"
#include "epilogue_params.cuh"
#include "philox.cuh"

namespace epi {

template <typename T, typename Act>
__global__ void kBiasActDropout(EpParams<T> p, Act actop) {
  using namespace epi;
  int64_t tid = blockDim.x * (int64_t)blockIdx.x + threadIdx.x;
  int64_t total = p.M * p.N;
  if (tid >= total) return;

  int64_t r = tid / p.N;
  int64_t c = tid % p.N;

  // load
  T x = p.x[r * p.ld_x + c];

  // bias + activation
  BiasAct<T, Act> f{p.bias, actop, p.N};
  T y = f(x, c);

  // dropout
  if (p.dropout_p > 0.f) {
    float u = rand01(p.seed, static_cast<uint64_t>(tid));

    float keep_prob = 1.f - p.dropout_p;
    uint8_t keep = (u < keep_prob);
    if (p.save_mask && p.mask) p.mask[r * p.N + c] = keep;
    float scale = keep ? (1.f / keep_prob) : 0.f;
    float yf = (keep ? 1.f : 0.f) * scale * pmath::to_f32(y);
    y = pmath::from_f32<T>(yf);
  } else {
    if (p.save_mask && p.mask) p.mask[r * p.N + c] = 1; // all-kept
  }

  p.y[r * p.ld_y + c] = y;
}

template <typename T>
cudaError_t launch_kernel(const Plan& plan, const Tensors& t) {
  auto p = make_params<T>(plan, t);
  dim3 grid = compute_grid(p.M, p.N);
  dim3 block = compute_block();

  switch (plan.attrs.act) {
    case ActKind::None: {
      kBiasActDropout<T, ActNone<T>><<<grid, block>>>(p, ActNone<T>{});
      break;
    }
    case ActKind::ReLU: {
      kBiasActDropout<T, ActReLU<T>><<<grid, block>>>(p, ActReLU<T>{});
      break;
    }
    case ActKind::GELU: {
      kBiasActDropout<T, ActGELU<T>><<<grid, block>>>(p, ActGELU<T>{});
      break;
    }
    default: return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

// explicit instantiation exported to launcher_policy.cu via extern
template cudaError_t launch_kernel<float>(const Plan&, const Tensors&);
template cudaError_t launch_kernel<half>(const Plan&, const Tensors&);

} // namespace epi
