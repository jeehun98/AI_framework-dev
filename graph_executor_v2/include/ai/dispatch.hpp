#pragma once
#include <map>
#include <tuple>        // std::tie
#include "ai/tensor.hpp"
#include "ai/op_schema.hpp"   // Device, DType, Layout, ActKind, GemmAttrs

namespace ai {

// ------------------------------
// 공통 타입
// ------------------------------
enum class OpKind { GEMM, GEMM_BWD };

using StreamHandle = void*;   // reinterpret_cast<cudaStream_t>

// 단순 상태 코드
enum class Status {
  Ok = 0,

  // 공통 범주
  Invalid        = 1,
  Unimplemented  = 2,
  RuntimeError   = 3,

  // 세부 오류
  DeviceMismatch = 100,
  DtypeMismatch  = 101,
  LayoutMismatch = 102,
  ShapeMismatch  = 103,
  StrideMismatch = 104,
  TransposeNotSupported = 105,

  MissingInput   = 110,
  MissingOutput  = 111,
  RegistryNotFound = 120,

  Unknown        = 999
};

// ==============================
// Forward (GEMM + bias + act)
// ==============================
// NOTE: Z_saved는 nullptr 가능. save_z 옵션일 때 상위가 버퍼를 넘김.
using KernelFn = Status(*)(const Tensor& A, const Tensor& B, const Tensor* Bias,
                           Tensor& Y, const GemmAttrs&, StreamHandle, Tensor* Z_saved);

struct OpKey {
  OpKind  kind;      // GEMM
  Device  dev;
  DType   dtype;
  Layout  layout;
  ActKind act;       // None/ReLU/LeakyReLU/GELU/Sigmoid/Tanh
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
  GemmAttrs     attrs;   // leaky_slope 포함
};

class OpRegistry {
public:
  static OpRegistry& inst();
  void     reg(const OpKey& key, KernelFn fn);
  KernelFn find_best(const OpQuery& q) const; // 초기 버전: 단순 키 매칭
private:
  std::map<OpKey, KernelFn> table_;
};

// ==============================
// Backward (GEMM + bias + act BWD)
//  입력: A,B,(C), gY, Z
//  출력: gA,gB,(gC),(gBias)
// ==============================

// NOTE: attrs는 GemmAttrs 재사용 (act, leaky_slope)
using GemmBwdFn = Status(*)(const Tensor& A, const Tensor& B, const Tensor* C,
                            const Tensor& gY, const Tensor& Z,
                            Tensor* gA, Tensor* gB, Tensor* gC, Tensor* gBias,
                            const GemmAttrs&, StreamHandle);

struct BwdOpKey {
  OpKind  kind;      // GEMM_BWD
  Device  dev;
  DType   dtype;
  Layout  layout;
  ActKind act;
  bool    with_bias;

  bool operator<(const BwdOpKey& o) const {
    return std::tie(kind,dev,dtype,layout,act,with_bias)
         < std::tie(o.kind,o.dev,o.dtype,o.layout,o.act,o.with_bias);
  }
};

struct BwdOpQuery {
  OpKind         kind;     // GEMM_BWD
  const Tensor&  A;
  const Tensor&  B;
  const Tensor*  C;        // nullable
  const Tensor&  gY;
  const Tensor&  Z;
  GemmAttrs      attrs;
};

class BwdOpRegistry {
public:
  static BwdOpRegistry& inst();
  void      reg(const BwdOpKey& key, GemmBwdFn fn);
  GemmBwdFn find_best(const BwdOpQuery& q) const; // 초기: 단순 키 매칭
private:
  std::map<BwdOpKey, GemmBwdFn> table_;
};

// ==============================
// 상위 진입점(디스패치 래퍼) 선언
//  - 정의는 src/dispatch/registry.cpp 에서
// ==============================
namespace ops {

// FWD: 확장 엔트리 (Z 저장 지원)
int gemm_run_ex(const Tensor& A, const Tensor& B, const Tensor* Bias,
                Tensor& Y, const GemmAttrs& attrs, StreamHandle s,
                Tensor* Z_saved);

// FWD: 기존 엔트리 (호환용, Z 저장 없음)
int gemm_run(const Tensor& A, const Tensor& B, const Tensor* Bias,
             Tensor& Y, const GemmAttrs& attrs, StreamHandle s);

// BWD: 새 엔트리
int gemm_bwd_run(const Tensor& A, const Tensor& B, const Tensor* C,
                 const Tensor& gY, const Tensor& Z,
                 Tensor* gA, Tensor* gB, Tensor* gC, Tensor* gBias,
                 const GemmAttrs& attrs, StreamHandle s);

} // namespace ops

} // namespace ai
