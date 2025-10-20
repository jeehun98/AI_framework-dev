#include "epilogue.h"
#include <cuda_runtime.h>

namespace epi {
extern cudaError_t launch_policy_f16(const Plan&, const Tensors&);
extern cudaError_t launch_policy_f32(const Plan&, const Tensors&);

cudaError_t run(const Plan& plan, const Tensors& t, DType dtype) {
  if (plan.rows <= 0 || plan.cols <= 0) return cudaErrorInvalidValue;
  switch (dtype) {
    case DType::F16: return launch_policy_f16(plan, t);
    case DType::F32: return launch_policy_f32(plan, t);
    default:         return cudaErrorInvalidValue;
  }
}
} // namespace epi
