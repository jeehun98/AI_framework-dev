#pragma once
#include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#include <map>
#include <tuple>
#include <mutex>

namespace ai {

enum class OpKind : uint8_t { GEMM=0, GEMM_BWD=1 };

using KernelFn = Status(*)(const Tensor& A, const Tensor& B, const Tensor* Bias,
                           Tensor& Y, const GemmAttrs&, StreamHandle, Tensor* Z_saved);

struct OpKey {
  OpKind  kind;
  Device  dev;
  DType   dtype;
  Layout  layout;
  bool    trans_a;
  bool    trans_b;
  ActKind act;
  bool    with_bias;
  bool    save_z;

  bool operator<(const OpKey& o) const {
    return std::tie(kind,dev,dtype,layout,trans_a,trans_b,act,with_bias,save_z)
         < std::tie(o.kind,o.dev,o.dtype,o.layout,o.trans_a,o.trans_b,o.act,o.with_bias,o.save_z);
  }
};

struct OpQuery {
  OpKind        kind;
  const Tensor& A;
  const Tensor& B;
  const Tensor* Bias;
  Tensor&       Y;
  GemmAttrs     attrs;
};

class OpRegistry {
public:
  static OpRegistry& inst() { static OpRegistry g; return g; }
  void reg(const OpKey& key, KernelFn fn) {
    std::lock_guard<std::mutex> g(mu_); table_[key] = fn;
  }
  KernelFn find_best(const OpQuery& q) const {
    OpKey key{
      q.kind, Device::CUDA, q.A.desc.dtype, q.A.desc.layout,
      q.attrs.trans_a, q.attrs.trans_b, q.attrs.act,
      q.Bias != nullptr, q.attrs.save_z
    };
    std::lock_guard<std::mutex> g(mu_);
    auto it = table_.find(key);
    return (it != table_.end()) ? it->second : nullptr;
  }
private:
  mutable std::mutex mu_;
  std::map<OpKey, KernelFn> table_;
};

using GemmBwdFn = Status(*)(const Tensor& A, const Tensor& B, const Tensor* C,
                            const Tensor& gY, const Tensor& Z,
                            Tensor* gA, Tensor* gB, Tensor* gC, Tensor* gBias,
                            const GemmAttrs&, StreamHandle);

struct BwdOpKey {
  OpKind  kind;
  Device  dev;
  DType   dtype;
  Layout  layout;
  bool    trans_a;
  bool    trans_b;
  ActKind act;
  bool    with_bias;

  bool operator<(const BwdOpKey& o) const {
    return std::tie(kind,dev,dtype,layout,trans_a,trans_b,act,with_bias)
         < std::tie(o.kind,o.dev,o.dtype,o.layout,o.trans_a,o.trans_b,o.act,o.with_bias);
  }
};

struct BwdOpQuery {
  OpKind         kind;
  const Tensor&  A;
  const Tensor&  B;
  const Tensor*  C;
  const Tensor&  gY;
  const Tensor&  Z;
  GemmAttrs      attrs;
};

class BwdOpRegistry {
public:
  static BwdOpRegistry& inst() { static BwdOpRegistry g; return g; }
  void reg(const BwdOpKey& key, GemmBwdFn fn) {
    std::lock_guard<std::mutex> g(mu_); table_[key] = fn;
  }
  GemmBwdFn find_best(const BwdOpQuery& q) const {
    BwdOpKey key{
      q.kind, Device::CUDA, q.A.desc.dtype, q.A.desc.layout,
      q.attrs.trans_a, q.attrs.trans_b, q.attrs.act,
      /*with_bias=*/ (q.C != nullptr)
    };
    std::lock_guard<std::mutex> g(mu_);
    auto it = table_.find(key);
    return (it != table_.end()) ? it->second : nullptr;
  }
private:
  mutable std::mutex mu_;
  std::map<BwdOpKey, GemmBwdFn> table_;
};

namespace ops {

inline Status gemm_run_ex(const Tensor& A, const Tensor& B, const Tensor* Bias,
                          Tensor& Y, const GemmAttrs& attrs, StreamHandle s,
                          Tensor* Z_saved);

inline Status gemm_run(const Tensor& A, const Tensor& B, const Tensor* Bias,
                       Tensor& Y, const GemmAttrs& attrs, StreamHandle s);

inline Status gemm_bwd_run(const Tensor& A, const Tensor& B, const Tensor* C,
                           const Tensor& gY, const Tensor& Z,
                           Tensor* gA, Tensor* gB, Tensor* gC, Tensor* gBias,
                           const GemmAttrs& attrs, StreamHandle s);

} // namespace ops
} // namespace ai
