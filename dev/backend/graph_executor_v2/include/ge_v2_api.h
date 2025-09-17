#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Keep old typedef for pointer passing from Python (capsule/int)
typedef uintptr_t ge2_uintptr;

// Minimal public API compatible layer
typedef struct {
  int M;
  int N;
  int K;
  int has_bias;  // 0/1
  int act;       // 0:none, 1:ReLU   (legacy)
} ge2_gemm_bias_act_params_t;

// f32 entry (now internally calls regemm_epilogue)
int ge2_launch_gemm_bias_act_f32(const ge2_uintptr* bufs, int n, void* stream);

// (Optional) fp16 tensorcore entry (stubbed for now, returns -99 if not implemented here)
int ge2_launch_gemm_bias_act_tc_f16(const ge2_uintptr* bufs, int n, void* stream);

#ifdef __cplusplus
}
#endif
