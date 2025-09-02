#include <string>
#include <unordered_map>
#include "ge_v2_api.h"

// 외부 커널 진입점 선언
extern "C" {
  int ge2_launch_gemm_bias_act_tc_f16(const ge2_uintptr*, int, void*);
  int ge2_launch_gemm_bias_act_f32   (const ge2_uintptr*, int, void*);
}

const std::unordered_map<std::string, ge2_kernel_fn>& ge_v2_kernel_table_raw() {
  static std::unordered_map<std::string, ge2_kernel_fn> tab = {
    {"gemm_bias_act_tc_f16", &ge2_launch_gemm_bias_act_tc_f16},
    {"gemm_bias_act_f32",    &ge2_launch_gemm_bias_act_f32},
  };
  return tab;
}

// f16(TC) 우선(성능상 이점), f32는 백업
const std::unordered_map<std::string, int>& ge_v2_capability_table_raw() {
  static std::unordered_map<std::string, int> caps = {
    {"GEMM_BIAS_ACT__gemm_bias_act_tc_f16", 90},
    {"GEMM_BIAS_ACT__gemm_bias_act_f32",    70},
  };
  return caps;
}
