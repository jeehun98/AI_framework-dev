#include <cuda_runtime.h>
#include "regemm/api.h"

namespace regemm {
// Forward decls implemented in the .cu files
void launch_gemm_bias_act_f32_smoke (const GemmBiasActParams& p, cudaStream_t s);
void launch_gemm_bias_act_f32_tiled (const GemmBiasActParams& p, cudaStream_t s);

// Simple policy: use tiled for medium+ sizes, smoke for tiny shapes.
void gemm_bias_act_f32(const GemmBiasActParams& p, cudaStream_t s) {
  const bool tiny = (p.M*p.N < 4096) || (p.K < 8);
  if (tiny) launch_gemm_bias_act_f32_smoke(p, s);
  else      launch_gemm_bias_act_f32_tiled(p, s);
}

} // namespace regemm
