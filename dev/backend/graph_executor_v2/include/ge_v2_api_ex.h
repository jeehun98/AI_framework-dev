#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------------------
// 공용 타입
// ----------------------------------------------------------------------------
typedef uintptr_t ge2_uintptr;

// Bias/Activation enum (확장)
typedef enum {
  GE2_BIAS_SCALAR = 0,  // 단일 스칼라(실제 사용처에서 의미상 broadcast)
  GE2_BIAS_PER_M  = 1,  // 행별(M) 브로드캐스트
  GE2_BIAS_PER_N  = 2,  // 열별(N) 브로드캐스트
} ge2_bias_kind_t;

typedef enum {
  GE2_ACT_NONE       = 0,
  GE2_ACT_RELU       = 1,
  GE2_ACT_LEAKY_RELU = 2,
  GE2_ACT_GELU       = 3,
  GE2_ACT_SIGMOID    = 4,
  GE2_ACT_TANH       = 5,
} ge2_act_kind_t;


// ----------------------------------------------------------------------------
// Forward(EX) 파라미터
//   D = act( alpha*(A@B) + beta*C + bias )
//   - 포인터들은 bufs[]로 전달, 여기 구조체엔 스칼라/stride/플래그만 둔다
//   - row-major 가정; lda/ldb/ldc/ldd==0 이면 내부에서 기본값(K/N/N/N) 사용
//   - Z stash(activation 직전 pre-activation) 저장 옵션 추가
// ----------------------------------------------------------------------------
typedef struct {
  int M, N, K;

  // strides (0 또는 음수면 row-major 기본값으로 내부에서 세팅)
  int lda, ldb, ldc, ldd;

  // epilogue 계수
  float alpha;   // 기본 1.0f
  float beta;    // 기본 0.0f

  // 선택 플래그
  int use_C;     // 0/1 (C 포인터 사용 여부)
  int has_bias;  // 0/1 (bias 포인터 사용 여부)

  // kinds
  ge2_bias_kind_t bias_kind;
  ge2_act_kind_t  act_kind;

  // (선택) LeakyReLU slope
  float leaky_slope; // 미지정 시 0.01f 등 기본값 사용

  // ---- NEW: Z stash (pre-activation 저장) ----
  // bufs[]에 Z 포인터가 오고, 여기서 저장 여부/stride만 지정
  int  save_preact;  // 1이면 Z에 pre-activation 기록
  int  ldZ;          // 0이면 ldd 사용(동일 레이아웃 권장)
} ge2_gemm_bias_act_params_ex_t;

// Forward(EX) 런처
// bufs 레이아웃(순서 준수):
//   bufs[0]=A, bufs[1]=B,
//   (use_C ? bufs[2]=C : 생략),
//   bufs[2 or 3]=D,
//   (has_bias ? 다음=bias : 생략),
//   (save_preact ? 다음=Z : 생략),
//   마지막 = &params_ex
//
// 예시:
//  - use_C=0, has_bias=0, save_preact=0:
//      [A,B,D,&params]
//  - use_C=1, has_bias=1, save_preact=1:
//      [A,B,C,D,bias,Z,&params]
int ge2_launch_gemm_bias_act_f32_ex(const ge2_uintptr* bufs, int n, void* stream);


// ----------------------------------------------------------------------------
// Backward(EX) 파라미터
//   수식:
//     Z = alpha*(A@B) + beta*C + bias
//     Y = act(Z)
//     입력: A, B, (C), gY, Z
//     출력: gA, gB, (gC), (gBias)
//   - gC = beta * gZ (C 사용 시)
//   - gBias:
//       SCALAR -> sum(gZ)
//       PER_M  -> sum(gZ, axis=1)  (size M)
//       PER_N  -> sum(gZ, axis=0)  (size N)
// ----------------------------------------------------------------------------
typedef struct {
  int M, N, K;

  // Forward 입력 strides (row-major; 0/음수면 내부 기본값 사용)
  int lda, ldb, ldc;   // A:[M,K], B:[K,N], C:[M,N]

  // Grad/보조 텐서 strides
  int ldgY;            // gY:[M,N]
  int ldZ;             // Z :[M,N]

  // 출력 grad strides (포인터가 None이면 무시)
  int ldgA;            // gA:[M,K]
  int ldgB;            // gB:[K,N]
  int ldgC;            // gC:[M,N]

  // 하이퍼파라미터 (forward와 일관)
  float alpha;
  float beta;
  ge2_bias_kind_t bias_kind;
  ge2_act_kind_t  act_kind;
  float leaky_slope;    // LeakyReLU 도함수용
} ge2_gemm_bias_act_bwd_params_t;

// Backward(EX) 런처
// bufs 레이아웃(순서 준수):
//   bufs[0]=A, bufs[1]=B,
//   (C 사용 시 bufs[2]=C),
//   bufs[2 or 3]=gY,
//   다음=Z,
//   다음=gA,
//   다음=gB,
//   (gC 필요 시 다음=gC),
//   (gBias 필요 시 다음=gBias),
//   마지막 = &params_bwd
//
// 예시:
//  - C 미사용, gC/gBias 미요청:
//      [A,B,gY,Z,gA,gB,&params]
//  - C 사용, gC/gBias 요청:
//      [A,B,C,gY,Z,gA,gB,gC,gBias,&params]
int ge2_launch_gemm_bias_act_bwd_f32_ex(const ge2_uintptr* bufs, int n, void* stream);

#ifdef __cplusplus
}
#endif
