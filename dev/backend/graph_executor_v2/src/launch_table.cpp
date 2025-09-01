#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

extern "C" {
  int ge2_launch_gemm_bias_act_tc_f16(const uintptr_t*, int, void*);
  int ge2_launch_gemm_bias_act_f32   (const uintptr_t*, int, void*);
}

using KernelFn = int(*)(const uintptr_t*, int, void*);

const std::unordered_map<std::string, KernelFn>& ge_v2_kernel_table_raw() {
  static std::unordered_map<std::string, KernelFn> tab = {
    {"gemm_bias_act_tc_f16", &ge2_launch_gemm_bias_act_tc_f16},
    {"gemm_bias_act_f32",    &ge2_launch_gemm_bias_act_f32},
  };
  return tab;
}

const std::unordered_map<std::string, int>& ge_v2_capability_table_raw() {
  static std::unordered_map<std::string, int> caps = {
    {"GEMM_BIAS_ACT__gemm_bias_act_tc_f16", 80},
    {"GEMM_BIAS_ACT__gemm_bias_act_f32",    50},
  };
  return caps;
}
