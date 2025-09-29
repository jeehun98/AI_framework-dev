#pragma once
// Standalone 모드: ai_core 없이 빌드할 때 쓰는 얇은 타입 정의
// BUILD_STANDALONE_OPS 가 정의되어 있을 때 활성화됨.

#ifdef BUILD_STANDALONE_OPS

#include <cstdint>
#include <vector>
#include <array>
#include <map>
#include <tuple>

namespace ai {

// ========================= 공통 상태/유틸 =========================
enum class Status : int {
  Ok = 0,

  // 공통 범주
  Invalid        = 1,
  Unimplemented  = 2,
  RuntimeError   = 3,

  // 세부 오류 코드
  DeviceMismatch = 100,
  DtypeMismatch  = 101,
  LayoutMismatch = 102,
  ShapeMismatch  = 103,
  StrideMismatch = 104,
  TransposeNotSupported = 105,

  MissingInput   = 110,
  RegistryNotFound = 120,

  Unknown        = 999
};

using StreamHandle = void*; // reinterpret_cast<cudaStream_t>

// ========================= 스칼라 타입/레이아웃 =========================
enum class Device { CPU, CUDA };
enum class DType  { F32, F16, BF16, I32, I8 };
enum class Layout { RowMajor, ColMajor };

// ========================= 텐서 정의 =========================
struct TensorDesc {
  DType dtype{DType::F32};
  Layout layout{Layout::RowMajor};
  std::vector<int64_t> shape;   // ex) [M,K] (요소 단위)
  std::vector<int64_t> stride;  // 요소 단위 stride (bytes 아님)

  int64_t dim(int i) const { return shape.at(static_cast<size_t>(i)); }
};

struct Tensor {
  void* data{nullptr};     // device pointer (CUDA 가정)
  TensorDesc desc{};
  Device device{Device::CUDA};
  int device_index{0};     // GPU id (CUDA)

  bool is_cuda() const { return device == Device::CUDA; }

  // 편의: [M,N], RowMajor, 연속
  bool is_contiguous_rowmajor_2d() const {
    if (desc.shape.size() != 2 || desc.layout != Layout::RowMajor) return false;
    return desc.stride.size() == 2 && desc.stride[1] == 1;
  }
};

// GEMM용 leading dimension 헬퍼
inline int64_t lda(const Tensor& A){ // [M,K]
  return (A.desc.layout==Layout::RowMajor) ? A.desc.stride[0] : A.desc.stride[1];
}
inline int64_t ldb(const Tensor& B){ // [K,N]
  return (B.desc.layout==Layout::RowMajor) ? B.desc.stride[0] : B.desc.stride[1];
}
inline int64_t ldd(const Tensor& D){ // [M,N]
  return (D.desc.layout==Layout::RowMajor) ? D.desc.stride[0] : D.desc.stride[1];
}

// ========================= 활성화/속성 =========================
enum class ActKind : uint8_t {
  None      = 0,
  ReLU      = 1,
  LeakyReLU = 2,
  GELU      = 3,
  Sigmoid   = 4,
  Tanh      = 5,
};

// regemm와 1:1 매핑되도록 유지
struct GemmAttrs {
  bool     trans_a{false};
  bool     trans_b{false};
  ActKind  act{ActKind::None};
  bool     with_bias{false};
  float    leaky_slope{0.01f}; // LeakyReLU 기본값 일치
};

// ========================= 오퍼 종류/레지스트리 키(선언만) =========================
// (Standalone에선 보통 직접 커널 호출이므로 실제 레지스트리 사용은 선택사항.
//  단, 동일 시그니처 유지·호환을 위해 선언만 제공)
enum class OpKind { GEMM, GEMM_BWD };

// 커널 함수 포인터(전방 선언)
using KernelFn = Status(*)(const Tensor& A, const Tensor& B, const Tensor* Bias,
                           Tensor& Y, const GemmAttrs&, StreamHandle);

struct OpKey {
  OpKind  kind;      // GEMM
  Device  dev;
  DType   dtype;
  Layout  layout;
  ActKind act;
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
  GemmAttrs     attrs;
};

// 전방 선언(정의는 통합 빌드에서 제공)
class OpRegistry {
public:
  static OpRegistry& inst();
  void     reg(const OpKey& key, KernelFn fn);
  KernelFn find_best(const OpQuery& q) const;
};

// ========================= Backward용 시그니처(선언) =========================
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
  GemmBwdFn find_best() const;
};

// ========================= 상위 디스패치 엔트리(시그니처 맞춤) =========================
namespace ops {

// FWD: 이미 사용 중인 엔트리(링크 오류 방지용 선언)
int gemm_run(const Tensor& A, const Tensor& B, const Tensor* Bias,
             Tensor& Y, const GemmAttrs& attrs, StreamHandle s);

// BWD: 새로 추가될 엔트리
int gemm_bwd_run(const Tensor& A, const Tensor& B, const Tensor* C,
                 const Tensor& gY, const Tensor& Z,
                 Tensor* gA, Tensor* gB, Tensor* gC, Tensor* gBias,
                 const GemmAttrs& attrs, StreamHandle s);

} // namespace ops

} // namespace ai

#else
  // 통합 빌드 경로: 원래 ai 헤더 사용 (정의/레지스트리/엔트리 포함)
  #include "ai/dispatch.hpp"
  #include "ai/tensor.hpp"
  #include "ai/op_schema.hpp"
#endif
