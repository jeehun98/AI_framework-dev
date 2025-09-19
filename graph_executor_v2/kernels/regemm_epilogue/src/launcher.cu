// src/launcher.cu
#include <cuda_runtime.h>
#include "regemm/api.h"

namespace regemm {

// -------- 기존 forward(legacy) --------
void launch_gemm_bias_act_f32_smoke (const GemmBiasActParams& p, cudaStream_t s);
void launch_gemm_bias_act_f32_tiled (const GemmBiasActParams& p, cudaStream_t s);

void gemm_bias_act_f32(const GemmBiasActParams& p, cudaStream_t s) {
  const bool tiny = (p.M * p.N < 4096) || (p.K < 8);
  if (tiny) launch_gemm_bias_act_f32_smoke(p, s);
  else      launch_gemm_bias_act_f32_tiled(p, s);
}


// -------- NEW: EX forward(Z-stash) --------
// 구현은 regemm_gemm_bias_act.cu 쪽에 있음
void launch_gemm_bias_act_f32_smoke_ex (const GemmBiasActParamsEx& p, cudaStream_t s);
void launch_gemm_bias_act_f32_tiled_ex (const GemmBiasActParamsEx& p, cudaStream_t s);

void gemm_bias_act_f32_ex(const GemmBiasActParamsEx& p, cudaStream_t s) {
  // 간단 휴리스틱: 작은 문제는 smoke, 그 외 tiled
  const bool tiny = (p.M * p.N < 4096) || (p.K < 8);
  if (tiny) launch_gemm_bias_act_f32_smoke_ex(p, s);
  else      launch_gemm_bias_act_f32_tiled_ex(p, s);
}


// -------- Backward(EX) --------
// 주의: 역전파 엔트리는 regemm_backward.cu에서 정의됨.
// api.h에 선언되어 있으므로 여기서 다시 정의/라우팅할 필요 없음.
//  void gemm_bias_act_bwd_f32(const GemmBiasActBwdParams& p, cudaStream_t s);

} // namespace regemm
