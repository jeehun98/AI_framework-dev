// launch_table.cpp
#include "ge_v2_api.h"
#include "ge_v2_api_ex.h"  // EX forward/backward declarations
#include <cstdio>
#include <string>

// CUDA stream은 void* 로 전달된다 (cudaStream_t 재해석). nullptr => default stream.
static_assert(sizeof(ge2_uintptr) == sizeof(void*),
              "ge2_uintptr must match pointer size (uintptr_t)");

// 간단한 bufs 개수 체크 유틸 (부족하면 stderr로 알리고 -1 반환 유도)
static inline bool need_count(const char* name, int have, int want) {
  if (have < want) {
    std::fprintf(stderr, "[GE2] '%s' needs >= %d bufs (got %d)\n", name, want, have);
    return true;
  }
  return false;
}

// 바이트 정렬/널 여부 등 간단한 디버그 검증(디버그 빌드에서만)
#ifdef DEBUG
static inline bool check_not_null(const char* name, int idx, ge2_uintptr p) {
  if (p == 0) {
    std::fprintf(stderr, "[GE2][%s] bufs[%d] must be non-null\n", name, idx);
    return false;
  }
  return true;
}
#endif

extern "C" int ge2_launch(const char* name, const ge2_uintptr* bufs, int n, void* stream) {
  if (!name) return -1;

  const std::string k(name);

  // ------------------------------------------------------------------
  // 레거시 경로 (기존 유지)
  // ------------------------------------------------------------------
  if (k == "gemm_bias_act_f32") {
    // 최소 인자 수 방어적 확인 (기존 파서가 n을 유연히 처리한다면 완화 가능)
    if (need_count(name, n, 4)) return -1;  // 대략 [A,B,D,&params] 이상
    return ge2_launch_gemm_bias_act_f32(bufs, n, stream);
  }
  if (k == "gemm_bias_act_tc_f16") {
    if (need_count(name, n, 4)) return -1;
    return ge2_launch_gemm_bias_act_tc_f16(bufs, n, stream);
  }

  // ------------------------------------------------------------------
  // NEW: 확장 Forward (Z stash 지원)
  //
  //   bufs 레이아웃 (가변 길이):
  //     [0]=A, [1]=B, [2]=(C, if use_C), [3 or 4]=D,
  //     (bias if has_bias), (Z if save_preact), [last]=&params_ex
  //
  //   * forward에서는 params_ex.ldZ==0이면 커널이 ldZ=ldd로 해석함
  // ------------------------------------------------------------------
  if (k == "gemm_bias_act_f32_ex" || k == "gemm_bias_act_ex") { // alias 허용
    // 옵션이 모두 꺼진 경우 최소: [A,B,D,&params] → 4개
    if (need_count(name, n, 4)) return -1;

    // 여기서는 bufs 인덱스를 가변적으로 해석하므로 추가 검사는 런처 내부 파서에 위임
    return ge2_launch_gemm_bias_act_f32_ex(bufs, n, stream);
  }

  // ------------------------------------------------------------------
  // NEW: Backward (EX, epilogue-fused)
  //
  //   bufs 레이아웃 (고정 길이 10 슬롯):
  //     [0]=A, [1]=B, [2]=C (없으면 0),
  //     [3]=gY, [4]=Z,
  //     [5]=gA, [6]=gB,
  //     [7]=gC (없으면 0), [8]=gBias (없으면 0),
  //     [9]=&params_bwd
  //
  //   * Backward에서는 ldZ를 반드시 명시해야 함 (forward의 ldZ==0 규칙 없음)
  // ------------------------------------------------------------------
  if (k == "gemm_bias_act_bwd_f32_ex" || k == "gemm_bias_act_bwd") { // alias 허용
    if (need_count(name, n, 10)) return -1;

#ifdef DEBUG
    // 필수 포인터 빠른 검증: A,B,gY,Z,gA,gB,&params_bwd
    if (!check_not_null(name, 0, bufs[0])) return -1; // A
    if (!check_not_null(name, 1, bufs[1])) return -1; // B
    if (!check_not_null(name, 3, bufs[3])) return -1; // gY
    if (!check_not_null(name, 4, bufs[4])) return -1; // Z (pre-activation)
    if (!check_not_null(name, 5, bufs[5])) return -1; // gA
    if (!check_not_null(name, 6, bufs[6])) return -1; // gB
    if (!check_not_null(name, 9, bufs[9])) return -1; // &params_bwd
#endif
    return ge2_launch_gemm_bias_act_bwd_f32_ex(bufs, n, stream);
  }

  std::fprintf(stderr, "[GE2] unknown kernel: %s\n", name);
  return -1;
}
