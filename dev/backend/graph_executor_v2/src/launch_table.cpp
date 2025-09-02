/**
 * @file launch_table.cpp
 * @brief 커널 이름 -> 커널 진입점(래퍼) 매핑과 capability 스코어 테이블
 *
 * 이 파일은 네이티브 빌드 산물 안에서 "정적 심볼" 형태로 유지됩니다.
 * 파이썬 바인딩은 해당 심볼들을 참조하여 커널을 선택/실행합니다.
 */

#include <string>
#include <unordered_map>
#include "ge_v2_api.h"

// 외부 커널 런처 래퍼들 (C 링크)
extern "C" {
  // 예시: GEMM+BIAS+ACT 의 두 변형
  int ge2_launch_gemm_bias_act_tc_f16(const ge2_uintptr*, int, void*);
  int ge2_launch_gemm_bias_act_f32   (const ge2_uintptr*, int, void*);
}

// --- 커널 이름 -> 함수 포인터 테이블 -----------------------------------

const std::unordered_map<std::string, ge2_kernel_fn>& ge_v2_kernel_table_raw() {
  static std::unordered_map<std::string, ge2_kernel_fn> tab = {
    // key 는 파이썬 selector 에서 사용하는 km["name"] 과 정확히 일치해야 합니다.
    {"gemm_bias_act_tc_f16", &ge2_launch_gemm_bias_act_tc_f16},
    {"gemm_bias_act_f32",    &ge2_launch_gemm_bias_act_f32},
  };
  return tab;
}

// --- capability 스코어 테이블 -------------------------------------------

const std::unordered_map<std::string, int>& ge_v2_capability_table_raw() {
  static std::unordered_map<std::string, int> caps = {
    // key 형식: "<OPTYPE>__<KERNEL_NAME>"
    {"GEMM_BIAS_ACT__gemm_bias_act_tc_f16", 80},
    {"GEMM_BIAS_ACT__gemm_bias_act_f32",    50},
  };
  return caps;
}
