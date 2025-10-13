// src/dispatch/selector_rules.cpp
#include "ai/dispatch.hpp"

namespace ai {

// ---------------------------------------------------------------------
// 현재 규칙:
//  - FWD: OpQuery를 약간 정규화(with_bias 보정) 후 OpRegistry로 전달
//  - BWD: 그대로 BwdOpRegistry로 전달
//  - 추후 여기서 아키/사이즈/타일링 휴리스틱을 추가하면 됨
// ---------------------------------------------------------------------

static inline bool has_bias_data(const Tensor* bias) {
  return bias && bias->data;
}

// 선택 규칙: Forward (GEMM)
KernelFn select_gemm_kernel(const OpQuery& q_in) {
  // with_bias를 Bias 텐서 존재 유무로 정규화
  OpQuery q = q_in;
  q.attrs.with_bias = has_bias_data(q.Bias);

  // 필요시 여기서 추가 규칙:
  //  - 행렬 크기(M,N,K)에 따른 커널군 선택
  //  - 디바이스 특성(SM/아키) 기반 룰
  //  - save_z 여부(q.attrs.save_z) 등
  // 지금은 단순히 레지스트리로 위임
  return OpRegistry::inst().find_best(q);
}

// 선택 규칙: Backward (GEMM_BWD)
GemmBwdFn select_gemm_bwd_kernel(const BwdOpQuery& q) {
  // 추후 규칙(예: act 별 특화 커널, bias 축 형태 등)을 넣을 자리
  return BwdOpRegistry::inst().find_best(q);
}

} // namespace ai
