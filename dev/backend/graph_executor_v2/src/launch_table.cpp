#include "ge_v2_api.h"
#include <cstdio>
#include <string>

// Example of a trivial table; keep for compatibility with existing code
extern "C" int ge2_launch(const char* name, const ge2_uintptr* bufs, int n, void* stream) {
  if (!name) return -1;
  if (std::string(name) == "gemm_bias_act_f32") {
    return ge2_launch_gemm_bias_act_f32(bufs, n, stream);
  }
  if (std::string(name) == "gemm_bias_act_tc_f16") {
    return ge2_launch_gemm_bias_act_tc_f16(bufs, n, stream);
  }
  std::fprintf(stderr, "[GE2] unknown kernel: %s\n", name);
  return -1;
}
