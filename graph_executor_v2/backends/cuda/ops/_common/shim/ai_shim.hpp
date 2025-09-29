#pragma once

// CUDA 런처/커널에서 코어 없이도 컴파일할 수 있도록 하는 "얇은" shim.
// 기본(통합 빌드) 경로에선 절대 재선언하지 않고, 진짜 ai 헤더만 include.

#ifdef BUILD_STANDALONE_OPS

  #include <cstdint>
  #include <vector>
  #include <tuple>
  #include <map>

  namespace ai {

  // ---------- 공통 상태/유틸 ----------
  enum class Status : int {
    Ok = 0,
    Invalid = 1,
    Unimplemented = 2,
    RuntimeError = 3,
    DeviceMismatch = 100,
    DtypeMismatch  = 101,
    LayoutMismatch = 102,
    ShapeMismatch  = 103,
    StrideMismatch = 104,
    TransposeNotSupported = 105,
    MissingInput   = 110,
    RegistryNotFound = 120,
    Unknown = 999
  };

  using StreamHandle = void*; // reinterpret_cast<cudaStream_t>

  // ---------- 스칼라/레이아웃 ----------
  enum class Device { CPU, CUDA };
  enum class DType  { F32, F16, BF16, I32, I8 };
  enum class Layout { RowMajor, ColMajor };

  // ---------- 텐서 ----------
  struct TensorDesc {
    DType dtype{DType::F32};
    Layout layout{Layout::RowMajor};
    std::vector<int64_t> shape;   // [M,K] 등
    std::vector<int64_t> stride;  // 요소 단위 stride

    int64_t dim(int i) const { return shape.at(static_cast<size_t>(i)); }
  };

  struct Tensor {
    void* data{nullptr};         // device ptr (CUDA)
    TensorDesc desc{};
    Device device{Device::CUDA};
    int device_index{0};

    bool is_cuda() const { return device == Device::CUDA; }
    bool is_contiguous_rowmajor_2d() const {
      if (desc.shape.size() != 2 || desc.layout != Layout::RowMajor) return false;
      return desc.stride.size() == 2 && desc.stride[1] == 1;
    }
  };

  inline int64_t lda(const Tensor& A){
    return (A.desc.layout==Layout::RowMajor) ? A.desc.stride[0] : A.desc.stride[1];
  }
  inline int64_t ldb(const Tensor& B){
    return (B.desc.layout==Layout::RowMajor) ? B.desc.stride[0] : B.desc.stride[1];
  }
  inline int64_t ldd(const Tensor& D){
    return (D.desc.layout==Layout::RowMajor) ? D.desc.stride[0] : D.desc.stride[1];
  }

  // ---------- 활성화/속성 ----------
  enum class ActKind : uint8_t { None=0, ReLU=1, LeakyReLU=2, GELU=3, Sigmoid=4, Tanh=5 };

  struct GemmAttrs {
    bool     trans_a{false};
    bool     trans_b{false};
    ActKind  act{ActKind::None};
    bool     with_bias{false};
    float    leaky_slope{0.01f};
  };

  // ---------- 오퍼/레지스트리 키 ----------
  enum class OpKind { GEMM, GEMM_BWD };

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

  // 선언만 제공 (정의는 통합 빌드에서)
  class OpRegistry {
  public:
    static OpRegistry& inst();
    void     reg(const OpKey& key, KernelFn fn);
    KernelFn find_best(const OpQuery& q) const;
  };

  // ---------- Backward ----------
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

  // ---------- 상위 디스패치 엔트리(시그니처만) ----------
  namespace ops {
    int gemm_run(const Tensor& A, const Tensor& B, const Tensor* Bias,
                 Tensor& Y, const GemmAttrs& attrs, StreamHandle s);

    int gemm_bwd_run(const Tensor& A, const Tensor& B, const Tensor* C,
                     const Tensor& gY, const Tensor& Z,
                     Tensor* gA, Tensor* gB, Tensor* gC, Tensor* gBias,
                     const GemmAttrs& attrs, StreamHandle s);
  } // namespace ops

  } // namespace ai

#else  // 통합 빌드: 코어 공개 헤더만 사용 (재선언 금지)
  #include "ai/dispatch.hpp"
  #include "ai/tensor.hpp"
  #include "ai/op_schema.hpp"
#endif
