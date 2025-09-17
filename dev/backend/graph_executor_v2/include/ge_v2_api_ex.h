#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uintptr_t ge2_uintptr;

// Bias/Activation enum (확장)
typedef enum {
  GE2_BIAS_SCALAR = 0,  // 단일 스칼라(현재 커널에선 의미상 '없음'과 유사)
  GE2_BIAS_PER_M  = 1,  // 행별(M) 브로드캐스트
  GE2_BIAS_PER_N  = 2,  // 열별(N) 브로드캐스트
} ge2_bias_kind_t;

typedef enum {
  GE2_ACT_NONE      = 0,
  GE2_ACT_RELU      = 1,
  GE2_ACT_LEAKY_RELU= 2,
  GE2_ACT_GELU      = 3,
  GE2_ACT_SIGMOID   = 4,
  GE2_ACT_TANH      = 5,
} ge2_act_kind_t;

// 확장 파라미터 (α, β, C, stride/LD까지 노출)
typedef struct {
  int M, N, K;

  // strides (0 or 음수면 row-major 기본값으로 내부에서 세팅)
  int lda, ldb, ldc, ldd;

  // epilogue 계수
  float alpha;   // 기본 1.0f
  float beta;    // 기본 0.0f

  // 선택 플래그
  int use_C;     // 0/1 (C 포인터 유효 여부)
  int has_bias;  // 0/1 (bias 포인터 유효 여부)

  // kinds
  ge2_bias_kind_t bias_kind;
  ge2_act_kind_t  act_kind;

  // (선택) LeakyReLU slope 등 확장용
  float leaky_slope; // 미사용이면 0.01f 등 기본값, 현재 regemm는 고정/무시 가능
} ge2_gemm_bias_act_params_ex_t;

// 확장 엔트리: bufs 레이아웃
//  bufs[0]=A, bufs[1]=B, (use_C? bufs[2]=C), bufs[2 or 3]=D, (has_bias? 다음=bias), 마지막=&params_ex
//  즉, use_C/has_bias 조합에 따라 개수가 달라짐
int ge2_launch_gemm_bias_act_f32_ex(const ge2_uintptr* bufs, int n, void* stream);

#ifdef __cplusplus
}
#endif
