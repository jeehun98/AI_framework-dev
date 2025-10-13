// src/dispatch/registry.cpp
#include "ai/dispatch.hpp"

namespace ai
{

// ===========================
// Forward registry
// ===========================
OpRegistry& OpRegistry::inst() {
  static OpRegistry R; return R;
}
void OpRegistry::reg(const OpKey& k, KernelFn fn) { table_[k] = fn; }

static OpKey make_key_from(const OpQuery& q) {
  OpKey k{};
  k.kind      = q.kind;
  k.dev       = q.A.is_cuda() ? Device::CUDA : Device::CPU;
  k.dtype     = q.A.desc.dtype;
  k.layout    = q.Y.desc.layout;
  k.act       = q.attrs.act;
  k.with_bias = (q.Bias && q.Bias->data);
  return k;
}

KernelFn OpRegistry::find_best(const OpQuery& q) const {
  const auto key = make_key_from(q);

  if (auto it = table_.find(key); it != table_.end()) return it->second;

  auto k2 = key; k2.act = ActKind::None;
  if (auto it = table_.find(k2); it != table_.end()) return it->second;

  auto k3 = k2; k3.layout = Layout::RowMajor;
  if (auto it = table_.find(k3); it != table_.end()) return it->second;

  return nullptr;
}

// ===========================
// Backward registry
// ===========================
BwdOpRegistry& BwdOpRegistry::inst() {
  static BwdOpRegistry R; return R;
}
void BwdOpRegistry::reg(const BwdOpKey& k, GemmBwdFn fn) { table_[k] = fn; }

static BwdOpKey make_bwd_key_from(const BwdOpQuery& q) {
  BwdOpKey k{};
  k.kind      = q.kind;                                   // GEMM_BWD
  k.dev       = q.A.is_cuda() ? Device::CUDA : Device::CPU;
  k.dtype     = q.A.desc.dtype;                           // A 기준
  k.layout    = q.Z.desc.layout;                          // Z 레이아웃 기준
  k.act       = q.attrs.act;
  k.with_bias = q.attrs.with_bias;                        // FWD에서 bias 사용 여부
  return k;
}

// ★ 여기! 인자 없는 find_best() 구현을 지우고, 아래 시그니처로 교체
GemmBwdFn BwdOpRegistry::find_best(const BwdOpQuery& q) const {
  const auto key = make_bwd_key_from(q);

  if (auto it = table_.find(key); it != table_.end()) return it->second;

  auto k2 = key; k2.act = ActKind::None;
  if (auto it = table_.find(k2); it != table_.end()) return it->second;

  auto k3 = k2; k3.layout = Layout::RowMajor;
  if (auto it = table_.find(k3); it != table_.end()) return it->second;

  return nullptr;
}

// ===========================
// 상위 디스패치 진입점
// ===========================
namespace ops {

// FWD 공개 엔트리포인트는 src/ops/gemm.cpp에 있음

int gemm_bwd_run(const Tensor& A, const Tensor& B, const Tensor* C,
                 const Tensor& gY, const Tensor& Z,
                 Tensor* gA, Tensor* gB, Tensor* gC, Tensor* gBias,
                 const GemmAttrs& attrs, StreamHandle s)
{
  // 변경: 쿼리 만들어서 find_best(q) 호출
  BwdOpQuery q{ OpKind::GEMM_BWD, A, B, C, gY, Z, attrs };
  ai::GemmBwdFn fn = ai::BwdOpRegistry::inst().find_best(q);
  if (!fn) return -100; // not found

  ai::Status st = fn(A, B, C, gY, Z, gA, gB, gC, gBias, attrs, s);
  return (st == ai::Status::Ok) ? 0 : -7;
}

} // namespace ops

} // namespace ai
