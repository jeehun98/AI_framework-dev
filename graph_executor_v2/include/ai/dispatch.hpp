#pragma once
#include <map>
#include <tuple>        // for std::tie
#include "ai/tensor.hpp"
#include "ai/op_schema.hpp"   // ← ActKind, GemmAttrs 정의는 여기서만!

namespace ai {

enum class OpKind { GEMM };

using StreamHandle = void*;   // cudaStream_t 재해석
using Status = int;           // 0=OK, <0 error

// GEMM 커널 프로토타입
using KernelFn = Status(*)(const Tensor& A, const Tensor& B, const Tensor* Bias,
                           Tensor& Y, const GemmAttrs&, StreamHandle);

struct OpKey {
  OpKind  kind;
  Device  dev;
  DType   dtype;
  Layout  layout;
  ActKind act;       // ← ai::ActKind (op_schema.hpp에서 확장됨: LeakyReLU, Sigmoid 포함)
  bool    with_bias;

  bool operator<(const OpKey& o) const {
    return std::tie(kind,dev,dtype,layout,act,with_bias)
         < std::tie(o.kind,o.dev,o.dtype,o.layout,o.act,o.with_bias);
  }
};

struct OpQuery {
  OpKind        kind;
  const Tensor& A;
  const Tensor& B;
  const Tensor* Bias;
  Tensor&       Y;
  GemmAttrs     attrs;   // ← leaky_slope 포함
};

class OpRegistry {
public:
  static OpRegistry& inst();
  void     reg(const OpKey& key, KernelFn fn);
  KernelFn find_best(const OpQuery& q) const; // 매우 단순한 매칭
private:
  std::map<OpKey, KernelFn> table_;
};

} // namespace ai
