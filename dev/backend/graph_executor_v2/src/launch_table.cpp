#include <string>
#include <unordered_map>
#include "ge_v2_api.h"

/**
 * 커널 등록 테이블과 capability 점수 테이블
 * - 이름은 파이썬 selector의 km["name"] 과 정확히 일치해야 함
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
// (현재는 f32를 기본 smoke용으로 우선 선택되도록 점수 설정)
const std::unordered_map<std::string, int>& ge_v2_capability_table_raw() {
  static std::unordered_map<std::string, int> caps = {
    {"GEMM_BIAS_ACT__gemm_bias_act_tc_f16", 40},
    {"GEMM_BIAS_ACT__gemm_bias_act_f32",    80},
  };
  return caps;
}
