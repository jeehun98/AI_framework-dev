#include "ge_v2_api.h"
#include "ge_v2_api_ex.h"  // ✅ EX forward/backward 선언
#include <cstdio>
#include <string>

// Example of a trivial table; keep for compatibility with existing code
extern "C" int ge2_launch(const char* name, const ge2_uintptr* bufs, int n, void* stream) {
  if (!name) return -1;

  const std::string k(name);

  // ---- 레거시 경로 (기존 유지) ----
  if (k == "gemm_bias_act_f32") {
    return ge2_launch_gemm_bias_act_f32(bufs, n, stream);
  }
  if (k == "gemm_bias_act_tc_f16") {
    return ge2_launch_gemm_bias_act_tc_f16(bufs, n, stream);
  }

  // ---- NEW: 확장 Forward (Z stash 지원) ----
  // bufs: [A,B,(C),D,(bias),(Z), &params_ex]
  if (k == "gemm_bias_act_f32_ex" || k == "gemm_bias_act_ex") { // alias 허용
    return ge2_launch_gemm_bias_act_f32_ex(bufs, n, stream);
  }

  // ---- NEW: Backward (EX) ----
  // bufs: [A,B,(C), gY, Z, gA, gB, (gC), (gBias), &params_bwd]
  if (k == "gemm_bias_act_bwd_f32_ex" || k == "gemm_bias_act_bwd") { // alias 허용
    return ge2_launch_gemm_bias_act_bwd_f32_ex(bufs, n, stream);
  }

  std::fprintf(stderr, "[GE2] unknown kernel: %s\n", name);
  return -1;
}
