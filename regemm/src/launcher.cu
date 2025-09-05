#include <cuda_runtime.h>
#include "regemm/api.h"

namespace regemm {
void launch_gemm_bias_act_f32_smoke(const GemmBiasActParams&, cudaStream_t);
void launch_gemm_bias_act_f32_tiled(const GemmBiasActParams&, cudaStream_t);

int gemm_bias_act(const GemmBiasActParams& p, void* stream) {
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  if (p.dtype == DType::F32) {
    // 간단한 휴리스틱: 큰 문제는 tiled, 아주 작으면 smoke
    bool use_tiled = (p.M >= 128 && p.N >= 128 && p.K >= 16);
    if (use_tiled) launch_gemm_bias_act_f32_tiled(p, s);
    else           launch_gemm_bias_act_f32_smoke(p, s);
    return cudaPeekAtLastError();
  }
  return 1; // unsupported dtype
}
} // namespace regemm
