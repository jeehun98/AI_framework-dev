#pragma once
#include <cstdint>

// 레거시 ABI 호환용
using ge2_uintptr = uintptr_t;

extern "C" {
  // 당신의 레거시 심볼 목록 중 실제 쓰는 것만 선언
  int ge2_launch_gemm_bias_act_f32(const ge2_uintptr* bufs, int n, void* stream);
  int ge2_launch_gemm_bias_act_tc_f16(const ge2_uintptr* bufs, int n, void* stream);
  int ge2_launch_gemm_bias_act_f32_ex(const ge2_uintptr* bufs, int n, void* stream);
  int ge2_launch_gemm_bias_act_bwd_f32_ex(const ge2_uintptr* bufs, int n, void* stream);
}
