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
  Invalid        = 1,   // 일반 invalid
  Unimplemented  = 2,
  RuntimeError   = 3,

  // === 세부 오류 코드 (Backward GEMM 예시) ===
  DeviceMismatch = 100, // 디바이스 타입 불일치 (CPU/GPU 혼용 등)
  DtypeMismatch  = 101, // 데이터 타입 불일치 (예: F32만 지원인데 다른 dtype)
  LayoutMismatch = 102, // 레이아웃 불일치 (RowMajor만 지원인데 ColMajor 등)
  ShapeMismatch  = 103, // 텐서 shape 불일치
  StrideMismatch = 104, // stride가 예상과 맞지 않음
  TransposeNotSupported = 105, // trans_a, trans_b 아직 미지원

  MissingInput   = 110, // gC 요청했는데 C 없음 등
  RegistryNotFound = 120, // 레지스트리에서 커널 못 찾음

  Unknown        = 999
};
// ==============================
// Forward (GEMM + bias + act)
// ==============================
using KernelFn = Status(*)(const Tensor& A, const Tensor& B, const Tensor* Bias,
                           Tensor& Y, const GemmAttrs&, StreamHandle);

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
  KernelFn find_best(const OpQuery& q) const; // 매우 단순 매칭(초기 버전)
private:
  std::map<OpKey, KernelFn> table_;
};

// ==============================
// Backward (GEMM + bias + act BWD)
//  입력: A,B,(C), gY, Z
//  출력: gA,gB,(gC),(gBias)
// ==============================

// NOTE: attrs는 일단 GemmAttrs 재사용( act, leaky_slope )
//       필요시 alpha/beta 포함 전용 GemmBwdAttrs로 확장 가능.
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

class BwdOpRegistry {
public:
  static BwdOpRegistry& inst();
  void      reg(const BwdOpKey& key, GemmBwdFn fn);
  // 초기에는 단순 구현이라 인자 없이도 OK: 첫 엔트리 등 반환
  GemmBwdFn find_best() const;
private:
  std::map<BwdOpKey, GemmBwdFn> table_;
};

// ==============================
// 상위 진입점(디스패치 래퍼) 선언
//  - 정의는 src/dispatch/registry.cpp 에서
// ==============================
namespace ops {

// FWD: 이미 사용 중인 엔트리 (링크 오류 방지용 선언)
int gemm_run(const Tensor& A, const Tensor& B, const Tensor* Bias,
             Tensor& Y, const GemmAttrs& attrs, StreamHandle s);

// BWD: 새로 추가될 엔트리
int gemm_bwd_run(const Tensor& A, const Tensor& B, const Tensor* C,
                 const Tensor& gY, const Tensor& Z,
                 Tensor* gA, Tensor* gB, Tensor* gC, Tensor* gBias,
                 const GemmAttrs& attrs, StreamHandle s);

} // namespace ops




} // namespace ai
