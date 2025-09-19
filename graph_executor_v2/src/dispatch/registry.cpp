#include "ai/dispatch.hpp"

namespace ai {

OpRegistry& OpRegistry::inst(){ static OpRegistry R; return R; }
void OpRegistry::reg(const OpKey& k, KernelFn fn){ table_[k] = fn; }

static OpKey make_key_from(const OpQuery& q){
  OpKey k{};
  k.kind      = q.kind;
  k.dev       = q.A.is_cuda()? Device::CUDA : Device::CPU;
  k.dtype     = q.A.desc.dtype;
  k.layout    = q.Y.desc.layout;
  k.act       = q.attrs.act;                 // ← LeakyReLU/Sigmoid도 정상 전달
  k.with_bias = (q.Bias != nullptr);
  return k;
}

KernelFn OpRegistry::find_best(const OpQuery& q) const{
  auto key = make_key_from(q);
  if (auto it = table_.find(key); it != table_.end()) return it->second;

  // fallback: act=None
  auto k2 = key; k2.act = ActKind::None;
  if (auto it = table_.find(k2); it != table_.end()) return it->second;

  // fallback: layout RowMajor
  auto k3 = k2; k3.layout = Layout::RowMajor;
  if (auto it = table_.find(k3); it != table_.end()) return it->second;

  return nullptr;
}

} // namespace ai
