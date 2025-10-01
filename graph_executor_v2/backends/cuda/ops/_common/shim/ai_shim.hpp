// backends/cuda/ops/_common/shim/ai_shim.hpp
#pragma once

// -----------------------------------------------------------------------------
// Standalone ops 빌드용 "얇은" 코어 대체(shim).
// 통합 빌드가 아닐 때(BUILD_STANDALONE_OPS)만 사용되며,
// 커널/런처에서 필요한 최소 타입/함수/매크로/유틸을 제공합니다.
// -----------------------------------------------------------------------------

#ifdef BUILD_STANDALONE_OPS

  #include <cstdint>
  #include <cstddef>
  #include <vector>
  #include <tuple>
  #include <map>
  #include <stdexcept>
  #include <type_traits>
  #include <cuda_runtime_api.h> // cudaStream_t, cudaMemcpy 등

  namespace ai {

  // ---------------- Common status / stream ----------------
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
    CUDA_ERROR = 130,
    Unknown = 999
  };

  using StreamHandle = void*; // reinterpret_cast<cudaStream_t>(s)

  // ---------------- Scalar / layout ----------------
  enum class Device { CPU, CUDA };
  enum class DType  { F32, F16, BF16, I32, I8 };
  enum class Layout { RowMajor, ColMajor };

  inline std::size_t dtype_size(DType dt) {
    switch (dt) {
      case DType::F32:  return 4;
      case DType::F16:  return 2;
      case DType::BF16: return 2;
      case DType::I32:  return 4;
      case DType::I8:   return 1;
      default:          return 0;
    }
  }

  // ---------------- Tensor ----------------
  struct TensorDesc {
    DType dtype{DType::F32};
    Layout layout{Layout::RowMajor};
    std::vector<int64_t> shape;   // dims
    std::vector<int64_t> stride;  // element strides

    int64_t dim(int i) const { return shape.at(static_cast<size_t>(i)); }
  };

  struct Tensor {
    void* data{nullptr};         // device ptr (CUDA)
    TensorDesc desc{};
    Device device{Device::CUDA};
    int device_index{0};

    // Basic predicates
    bool is_cuda() const { return device == Device::CUDA; }
    bool is_defined() const { return data != nullptr; }

    // Row-major contiguous 2D check
    bool is_contiguous_rowmajor_2d() const {
      if (desc.shape.size() != 2 || desc.layout != Layout::RowMajor) return false;
      if (desc.stride.size() != 2) return false;
      const int64_t rows = desc.shape[0];
      const int64_t cols = desc.shape[1];
      return (desc.stride[1] == 1) && (desc.stride[0] == cols) && (rows >= 0 && cols >= 0);
    }

    // Number of elements
    int64_t numel() const {
      if (desc.shape.empty()) return 0;
      int64_t n = 1;
      for (auto v : desc.shape) {
        if (v < 0) return 0; // guard
        n *= v;
      }
      return n;
    }

    // Byte size
    int64_t nbytes() const {
      return static_cast<int64_t>(numel()) * static_cast<int64_t>(dtype_size(desc.dtype));
    }

    // Raw data ptr
    void* data_ptr() { return data; }
    const void* data_ptr() const { return data; }

    // Typed data ptr (non-const)
    template <typename T>
    T* data_ptr() {
      static_assert(!std::is_const<T>::value, "Use const overload for const type");
      return reinterpret_cast<T*>(data);
    }

    // Typed data ptr (const)
    template <typename T>
    const T* data_ptr() const {
      return reinterpret_cast<const T*>(data);
    }
  };

  // Leading dimension helpers (row/col-major)
  inline int64_t lda(const Tensor& A){
    return (A.desc.layout==Layout::RowMajor) ? A.desc.stride[0] : A.desc.stride[1];
  }
  inline int64_t ldb(const Tensor& B){
    return (B.desc.layout==Layout::RowMajor) ? B.desc.stride[0] : B.desc.stride[1];
  }
  inline int64_t ldd(const Tensor& D){
    return (D.desc.layout==Layout::RowMajor) ? D.desc.stride[0] : D.desc.stride[1];
  }

  // Row-major strides utility
  inline std::vector<int64_t> make_rowmajor_strides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> s(shape.size());
    int64_t st = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      s[i] = st;
      st  *= shape[i];
    }
    return s;
  }

  // Convenience makers
  inline Tensor make_tensor2d(void* ptr, int64_t rows, int64_t cols) {
    Tensor t;
    t.data = ptr;
    t.device = Device::CUDA;
    t.device_index = 0;
    t.desc.dtype  = DType::F32;
    t.desc.layout = Layout::RowMajor;
    t.desc.shape  = {rows, cols};
    t.desc.stride = {cols, 1};
    return t;
  }

  inline Tensor make_tensor_from_ptr(void* ptr, const std::vector<int64_t>& shape) {
    Tensor t;
    t.data = ptr;
    t.device = Device::CUDA;
    t.device_index = 0;
    t.desc.dtype  = DType::F32;
    t.desc.layout = Layout::RowMajor;
    t.desc.shape  = shape;
    t.desc.stride = make_rowmajor_strides(shape);
    return t;
  }

  // ---------------- Activations / attrs ----------------
  enum class ActKind : uint8_t { None=0, ReLU=1, LeakyReLU=2, GELU=3, Sigmoid=4, Tanh=5 };

  struct GemmAttrs {
    bool     trans_a{false};
    bool     trans_b{false};
    ActKind  act{ActKind::None};
    bool     with_bias{false};
    float    leaky_slope{0.01f};
  };

  // ---------------- Registry stubs (declaration only) ----------------
  enum class OpKind { GEMM, GEMM_BWD };

  using KernelFn = Status(*)(const Tensor& A, const Tensor& B, const Tensor* Bias,
                             Tensor& Y, const GemmAttrs&, StreamHandle);

  struct OpKey {
    OpKind  kind;
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

  // Only declarations — implementation provided by core build.
  class OpRegistry {
  public:
    static OpRegistry& inst();
    void     reg(const OpKey& key, KernelFn fn);
    KernelFn find_best(const OpQuery& q) const;
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

  namespace ops {
    // Forward GEMM entry used by RNN launcher (provided by core)
    // Returns 0 on success, non-zero on failure
    int gemm_run(const Tensor& A, const Tensor& B, const Tensor* Bias,
                 Tensor& Y, const GemmAttrs& attrs, StreamHandle s);

    int gemm_bwd_run(const Tensor& A, const Tensor& B, const Tensor* C,
                     const Tensor& gY, const Tensor& Z,
                     Tensor* gA, Tensor* gB, Tensor* gC, Tensor* gBias,
                     const GemmAttrs& attrs, StreamHandle s);
  } // namespace ops

  // ---------------- Error propagation macros ----------------
  #ifndef AI_RETURN_IF_ERROR
  #define AI_RETURN_IF_ERROR(expr)                          \
    do {                                                    \
      ::ai::Status _st__ = (expr);                          \
      if (_st__ != ::ai::Status::Ok) return _st__;          \
    } while(0)
  #endif

  #ifndef AI_CUDA_TRY
  #define AI_CUDA_TRY(cuda_expr)                                            \
    do {                                                                    \
      cudaError_t _cerr__ = (cuda_expr);                                    \
      if (_cerr__ != cudaSuccess) return ::ai::Status::RuntimeError;        \
    } while(0)
  #endif

  } // namespace ai

#else  // BUILD_STANDALONE_OPS

  // 통합 빌드에서는 코어 헤더만 사용 (재선언 금지)
  #include "ai/dispatch.hpp"
  #include "ai/tensor.hpp"
  #include "ai/op_schema.hpp"

#endif // BUILD_STANDALONE_OPS
