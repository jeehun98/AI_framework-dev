#include "ai/dispatch.hpp"

namespace ai
{

// ===========================
// Forward registry (기존)
// ===========================
OpRegistry& OpRegistry::inst() {
  static OpRegistry R; return R;
}

void OpRegistry::reg(const OpKey& k, KernelFn fn) {
  table_[k] = fn;
}

static OpKey make_key_from(const OpQuery& q) {
  OpKey k{};
  k.kind      = q.kind;                                   // GEMM
  k.dev       = q.A.is_cuda() ? Device::CUDA : Device::CPU;
  k.dtype     = q.A.desc.dtype;
  k.layout    = q.Y.desc.layout;
  k.act       = q.attrs.act;                              // None/ReLU/LeakyReLU/...
  k.with_bias = (q.Bias && q.Bias->data);                 // 더 안전한 판정
  return k;
}

KernelFn OpRegistry::find_best(const OpQuery& q) const {
  const auto key = make_key_from(q);

  // 1) exact
  if (auto it = table_.find(key); it != table_.end()) return it->second;

  // 2) fallback: act=None
  auto k2 = key; k2.act = ActKind::None;
  if (auto it = table_.find(k2); it != table_.end()) return it->second;

  // 3) fallback: layout RowMajor
  auto k3 = k2; k3.layout = Layout::RowMajor;
  if (auto it = table_.find(k3); it != table_.end()) return it->second;

  return nullptr;
}

// ===========================
// Backward registry (신규)
// ===========================
BwdOpRegistry& BwdOpRegistry::inst() {
  static BwdOpRegistry R; return R;
}

void BwdOpRegistry::reg(const BwdOpKey& k, GemmBwdFn fn) {
  table_[k] = fn;
}

// 초기 버전: 아주 단순 — 등록된 첫 엔트리 반환
// (원하면 OpKey와 유사한 make_bwd_key_from(...)을 만들어 exact 매칭으로 확장 가능)
GemmBwdFn BwdOpRegistry::find_best() const {
  if (table_.empty()) return nullptr;
  return table_.begin()->second;
}

// ===========================
// 상위 디스패치 진입점 정의
// ===========================
namespace ops {

// FWD 공개 엔트리포인트는 src/ops/gemm.cpp에서 정의합니다.

// BWD: (신규)
int gemm_bwd_run(const Tensor& A, const Tensor& B, const Tensor* C,
                 const Tensor& gY, const Tensor& Z,
                 Tensor* gA, Tensor* gB, Tensor* gC, Tensor* gBias,
                 const GemmAttrs& attrs, StreamHandle s)
{
  // 현재는 매우 단순 레지스트리 → 첫 엔트리
  ai::GemmBwdFn fn = ai::BwdOpRegistry::inst().find_best();
  if (!fn) return -100; // not found

  ai::Status st = fn(A, B, C, gY, Z, gA, gB, gC, gBias, attrs, s);
  return (st == ai::Status::Ok) ? 0 : -7;  // 필요 시 매핑 테이블로 세분화
}


} // namespace ops

} // namespace ai
