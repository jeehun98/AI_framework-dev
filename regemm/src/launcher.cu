#include <cuda_runtime.h>
#include "regemm/api.h"

namespace regemm {
void launch_gemm_bias_act_f32_smoke(const GemmBiasActParams&, cudaStream_t);

int gemm_bias_act(const GemmBiasActParams& p, void* stream) {
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  if (p.dtype == DType::F32) {
    launch_gemm_bias_act_f32_smoke(p, s);
    return cudaPeekAtLastError();
  }
  // TODO: F16/BF16/WMMA 경로 선택
  return 1; // unsupported dtype
}
} // namespace regemm
