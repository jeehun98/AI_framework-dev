#include <cuda_runtime.h>
#include "../api/epilogue.h"

namespace epi {
cudaError_t dispatch_f32(const Plan&, const Tensors&);
cudaError_t dispatch_f16(const Plan&, const Tensors&);

// public entry (used by api/epilogue_stub.cpp)
cudaError_t launch_policy_f32(const Plan& plan, const Tensors& t) {
  return dispatch_f32(plan, t);
}
cudaError_t launch_policy_f16(const Plan& plan, const Tensors& t) {
  return dispatch_f16(plan, t);
}
} // namespace epi
