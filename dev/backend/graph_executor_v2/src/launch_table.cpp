#include <string>
#include <unordered_map>
#include "ge_v2_api.h"

/**
 * @file launch_table.cpp
 * @brief 커널 이름 → 함수 포인터 매핑 + capability 점수 테이블
 */

// 외부 커널 런처 래퍼(C 링크)
extern "C" {
  int ge2_launch_gemm_bias_act_tc_f16(const ge2_uintptr*, int, void*);
  int ge2_launch_gemm_bias_act_f32   (const ge2_uintptr*, int, void*);
}

// 커널 이름 -> 진입점
const std::unordered_map<std::string, ge2_kernel_fn>& ge_v2_kernel_table_raw() {
  static std::unordered_map<std::string, ge2_kernel_fn> tab = {
    {"gemm_bias_act_tc_f16", &ge2_launch_gemm_bias_act_tc_f16},
    {"gemm_bias_act_f32",    &ge2_launch_gemm_bias_act_f32},
  };
  return tab;
}

// "<OPTYPE>__<KERNEL_NAME>" -> score
const std::unordered_map<std::string, int>& ge_v2_capability_table_raw() {
  static std::unordered_map<std::string, int> caps = {
    {"GEMM_BIAS_ACT__gemm_bias_act_tc_f16", 80},
    {"GEMM_BIAS_ACT__gemm_bias_act_f32",    50},
  };
  return caps;
}
