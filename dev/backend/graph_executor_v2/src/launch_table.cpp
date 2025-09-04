// launch_table.cpp
//
// 역할
//  - 네이티브 커널 엔트리포인트(함수 포인터)들을 문자열 키로 매핑한다.
//  - Executor 쪽에서 pick_kernel()이 선택한 "커널이름"으로 여기 테이블을 조회,
//    해당 함수 포인터(ge2_kernel_fn)를 받아 launch 시 호출한다.
//  - capability 테이블은 "OPTYPE__KERNEL" → score 로, 선택 우선순위 힌트를 준다.
//    (높을수록 선호. 무조건적 규칙은 아니고, pick_kernel 구현에 따라 가중치로 쓰임)
//
// 주의
//  - ge_v2_api.h에는 아래 두 함수가 extern "C"로 선언되어 있음.
//    여기서도 동일한 C 링크 규약(이름 unmangled)을 지켜 주는 편이 안전하다.
//    (C++ 타입을 반환해도 링크 규약만 C로 맞추는 건 가능함; 양쪽 다 C++로 컴파일된다는 가정)

#include <string>
#include <unordered_map>
#include "ge_v2_api.h"

// ------------------------------ 외부 커널 진입점 선언 ------------------------------
// * 각 진입점은 ge_v2_api.h에서 정의된 시그니처와 동일해야 한다.
// * 그래프 실행기(Executor)가 bufs, num_buffers, stream을 넘긴다.
// * 여기서는 선언만 하고, 정의는 각각의 구현 파일(my_kernels.cu 등)에 있다.
extern "C" {
  int ge2_launch_gemm_bias_act_tc_f16(const ge2_uintptr*, int, void*);
  int ge2_launch_gemm_bias_act_f32   (const ge2_uintptr*, int, void*);
}

// ------------------------------ 커널 테이블 ------------------------------
// 반환형이 "const std::unordered_map<...>&" 인 이유:
//  - 정적 지역 객체(static)로 한 번만 생성해, 테이블을 "싱글톤"처럼 재사용.
//  - 복사 비용도 피하고, 모듈 로드시 Thread-safe 초기화 보장(C++11 이후).
extern "C"
const std::unordered_map<std::string, ge2_kernel_fn>& ge_v2_kernel_table_raw() {
  // key:   커널명(Executor가 launch 시 넘기는 문자열)
  // value: 해당 커널의 함수 포인터(ge2_kernel_fn)
  static std::unordered_map<std::string, ge2_kernel_fn> tab = {
    // FP16 TensorCore + cuBLASLt (Bias(+ReLU) 에필로그 Fuse 지원)
    {"gemm_bias_act_tc_f16", &ge2_launch_gemm_bias_act_tc_f16},
    // 디버그/백업 경로: 순수 f32 스모크(직접 GEMM 커널 + 후처리)
    {"gemm_bias_act_f32",    &ge2_launch_gemm_bias_act_f32},
  };
  return tab;
}

// ------------------------------ capability 테이블 ------------------------------
// 형식: "OPTYPE__KERNEL" -> score
//  - OPTYPE: IR/플래너가 생성한 논리 연산 타입 (예: GEMM_BIAS_ACT)
//  - KERNEL: 위 커널 테이블의 키 문자열과 동일해야 한다.
//  - score: 높을수록 선호. 하드 필터가 아니라 "힌트" 값(선택 정책은 상위 레이어).
extern "C"
const std::unordered_map<std::string, int>& ge_v2_capability_table_raw() {
  static std::unordered_map<std::string, int> caps = {
    // FP16 TensorCore 경로를 가장 우선(성능상 이점)
    {"GEMM_BIAS_ACT__gemm_bias_act_tc_f16", 90},
    // f32 경로는 백업/호환용으로 낮은 점수
    {"GEMM_BIAS_ACT__gemm_bias_act_f32",    70},
  };
  return caps;
}
