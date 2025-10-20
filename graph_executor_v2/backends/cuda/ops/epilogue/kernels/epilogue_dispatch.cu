#include <cuda_runtime.h>
#include "../api/epilogue.h"
#include "epilogue_params.cuh"

namespace epi {

// kernels (templated) are in epilogue_kernels_policy.cu
template <typename T>
cudaError_t launch_kernel(const Plan& plan, const Tensors& t);

static inline const char* act_name(ActKind a){
  switch(a){
    case ActKind::None: return "None";
    case ActKind::ReLU: return "ReLU";
    case ActKind::GELU: return "GELU";
  }
  return "Unknown";
}

static inline bool check_inputs(const Plan& plan, const Tensors& t) {
  if (!t.x || !t.y) return false;
  if (plan.rows <= 0 || plan.cols <= 0) return false;
  if (plan.attrs.dropout_p < 0.f || plan.attrs.dropout_p >= 1.f) return false;
  if (plan.attrs.save_mask && !t.mask_out) {
    // mask 저장이 필요한데 포인터가 없는 경우 허용(무시)할지 여부.
    // 여기서는 허용: 내부에서 nullptr 체크 후 skip 저장.
  }
  return true;
}

cudaError_t dispatch_f32(const Plan& plan, const Tensors& t){ return launch_kernel<float>(plan, t); }
cudaError_t dispatch_f16(const Plan& plan, const Tensors& t){ return launch_kernel<half>(plan, t); }

} // namespace epi
