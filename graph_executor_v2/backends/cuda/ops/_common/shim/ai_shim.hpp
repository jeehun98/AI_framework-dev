#pragma once

// -----------------------------------------------------------------------------
// Standalone ops 빌드용 "얇은" 코어 대체(shim).
// 통합 빌드가 아닐 때(BUILD_STANDALONE_OPS)만 사용되며,
// 커널/런처에서 필요한 최소 타입/함수/매크로/유틸을 제공합니다.
//
// 업데이트 요약(캡처-세이프 강화):
//  - CUDA Graph 캡처 상태 조회/가드 (host sync/malloc/동기 memcpy 금지)
//  - memcpy/memset async 래퍼, malloc/free 캡처 금지 래퍼
//  - 커널 런치 에러 수거 매크로(AI_CUDA_CHECK_LAUNCH)
//  - 입력 검증 유틸(2D row-major, Per-N bias 등)
//  - 스트림 핸들 캐스터(as_cuda_stream), dtype/numel 보강
//  - (옵션) NVTX 범위 태깅 매크로
// -----------------------------------------------------------------------------

#ifdef BUILD_STANDALONE_OPS

  #include <cstdint>
  #include <cstddef>
  #include <vector>
  #include <tuple>
  #include <map>
  #include <stdexcept>
  #include <type_traits>
  #include <limits>
  #include <cuda_runtime_api.h> // cudaStream_t, cudaMemcpy 등

  // (옵션) NVTX
  #ifdef AI_USE_NVTX
    #include "nvToolsExt.h"
  #endif

  namespace ai {

  // ---------------- Common status / stream ----------------
  enum class Status : int {
    Ok = 0,
    Invalid = 1,
    Unimplemented = 2,
    RuntimeError = 3,

    // 상세 오류
    DeviceMismatch = 100,
    DtypeMismatch  = 101,
    LayoutMismatch = 102,
    ShapeMismatch  = 103,
    StrideMismatch = 104,
    TransposeNotSupported = 105,

    MissingInput   = 110,
    MissingOutput  = 111,  // save_z 요청인데 Z가 없을 때 등
    RegistryNotFound = 120,

    CUDA_ERROR = 130,
    Unknown = 999
  };

  using StreamHandle = void*; // reinterpret_cast<cudaStream_t>(s)

  // 타입-세이프 스트림 변환
  inline cudaStream_t as_cuda_stream(StreamHandle s) {
    return reinterpret_cast<cudaStream_t>(s);
  }

  // 캡처 상태 조회
  enum class CapturePhase { None, Active, Invalid };

  inline CapturePhase get_capture_phase(StreamHandle s) {
    cudaStreamCaptureStatus st; unsigned long long id = 0;
    cudaError_t e = cudaStreamGetCaptureInfo_v2(as_cuda_stream(s), &st, &id, nullptr, nullptr);
    if (e != cudaSuccess) return CapturePhase::Invalid;
    if (st == cudaStreamCaptureStatusActive) return CapturePhase::Active;
    if (st == cudaStreamCaptureStatusNone)   return CapturePhase::None;
    return CapturePhase::Invalid;
  }

  // ---------------- Scalar / layout ----------------
  enum class Device { CPU, CUDA };
  enum class DType  { F32, F16, BF16, I32, I8 };
  enum class Layout { RowMajor, ColMajor };

  [[nodiscard]] inline constexpr std::size_t dtype_size(DType dt) {
    switch (dt) {
      case DType::F32:  return 4;
      case DType::F16:  return 2;
      case DType::BF16: return 2;
      case DType::I32:  return 4;
      case DType::I8:   return 1;
      default:          return 0;
    }
  }

  // ---------------- Size helpers ----------------
  inline int64_t safe_mul_nonneg(int64_t a, int64_t b) {
    if (a == 0 || b == 0) return 0;
    if (a > (std::numeric_limits<int64_t>::max() / b)) return -1; // overflow
    return a * b;
  }

  inline int64_t numel_of(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 0;
    int64_t n = 1;
    for (auto v : shape) {
      if (v < 0) return 0;
      n = safe_mul_nonneg(n, v);
      if (n < 0) return 0;
    }
    return n;
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
      return numel_of(desc.shape);
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
    if (A.desc.stride.size() != 2) return -1;
    return (A.desc.layout==Layout::RowMajor) ? A.desc.stride[0] : A.desc.stride[1];
  }
  inline int64_t ldb(const Tensor& B){
    if (B.desc.stride.size() != 2) return -1;
    return (B.desc.layout==Layout::RowMajor) ? B.desc.stride[0] : B.desc.stride[1];
  }
  inline int64_t ldd(const Tensor& D){
    if (D.desc.stride.size() != 2) return -1;
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
    bool     save_z{false}; // Z(pre-activation) 저장 의도
  };

  // ---------------- Registry stubs (declaration only) ----------------
  enum class OpKind { GEMM, GEMM_BWD };

  // FWD 커널 시그니처(Z_saved는 nullable)
  using KernelFn = Status(*)(const Tensor& A, const Tensor& B, const Tensor* Bias,
                             Tensor& Y, const GemmAttrs&, StreamHandle, Tensor* Z_saved);

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

  // BWD 커널
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
    GemmBwdFn find_best(const BwdOpQuery& q) const;
  };

  namespace ops {
    // Forward GEMM entry used by other modules (provided by core)
    // Returns 0 on success, non-zero on failure

    // Z 저장 지원 버전
    int gemm_run_ex(const Tensor& A, const Tensor& B, const Tensor* Bias,
                    Tensor& Y, const GemmAttrs& attrs, StreamHandle s,
                    Tensor* Z_saved);

    // 호환: 기존(Z 저장 없음)
    int gemm_run(const Tensor& A, const Tensor& B, const Tensor* Bias,
                 Tensor& Y, const GemmAttrs& attrs, StreamHandle s);

    int gemm_bwd_run(const Tensor& A, const Tensor& B, const Tensor* C,
                     const Tensor& gY, const Tensor& Z,
                     Tensor* gA, Tensor* gB, Tensor* gC, Tensor* gBias,
                     const GemmAttrs& attrs, StreamHandle s);
  } // namespace ops

  // ---------------- Capture-safe guards & wrappers ----------------

  // 캡처 중 금지: host sync, malloc/free, 동기 memcpy 등
  #ifndef AI_CAPTURE_FORBID_IF_ACTIVE
  #define AI_CAPTURE_FORBID_IF_ACTIVE(stream_handle, what)                           \
    do {                                                                             \
      ::ai::CapturePhase _ph = ::ai::get_capture_phase(stream_handle);               \
      if (_ph == ::ai::CapturePhase::Active) {                                       \
        (void)(what);                                                                \
        return ::ai::Status::RuntimeError;                                           \
      }                                                                              \
    } while(0)
  #endif

  // Async memcpy/memset 래퍼(항상 지정 스트림 사용)
  inline Status ai_memcpy_async(void* dst, const void* src, size_t nbytes,
                                cudaMemcpyKind kind, StreamHandle s) {
    if (get_capture_phase(s) == CapturePhase::Invalid) return Status::RuntimeError;
    cudaError_t e = cudaMemcpyAsync(dst, src, nbytes, kind, as_cuda_stream(s));
    return (e == cudaSuccess) ? Status::Ok : Status::RuntimeError;
  }

  inline Status ai_memset_async(void* dst, int value, size_t nbytes, StreamHandle s) {
    cudaError_t e = cudaMemsetAsync(dst, value, nbytes, as_cuda_stream(s));
    return (e == cudaSuccess) ? Status::Ok : Status::RuntimeError;
  }

  // 캡처 중 cudaMalloc/Free 금지 (워크스페이스 사전 준비 유도)
  inline Status ai_malloc(void** p, size_t nbytes, StreamHandle s) {
    AI_CAPTURE_FORBID_IF_ACTIVE(s, "cudaMalloc");
    cudaError_t e = cudaMalloc(p, nbytes);
    return (e == cudaSuccess) ? Status::Ok : Status::RuntimeError;
  }
  inline Status ai_free(void* p, StreamHandle s) {
    AI_CAPTURE_FORBID_IF_ACTIVE(s, "cudaFree");
    cudaError_t e = cudaFree(p);
    return (e == cudaSuccess) ? Status::Ok : Status::RuntimeError;
  }

  // 커널 런치 에러 수거(launch-time)
  #ifndef AI_CUDA_CHECK_LAUNCH
  #define AI_CUDA_CHECK_LAUNCH()                                                     \
    do {                                                                             \
      cudaError_t _e1 = cudaGetLastError();                                          \
      if (_e1 != cudaSuccess) return ::ai::Status::RuntimeError;                     \
    } while(0)
  #endif

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

  // ---------------- NVTX(옵션) ----------------
  #ifdef AI_USE_NVTX
    struct NvtxRange {
      const char* name;
      explicit NvtxRange(const char* n): name(n) { nvtxRangePushA(name); }
      ~NvtxRange(){ nvtxRangePop(); }
    };
    #define AI_NVTX_RANGE(name) ::ai::NvtxRange _nvtx_range__(name)
  #else
    #define AI_NVTX_RANGE(name) ((void)0)
  #endif

  // ---------------- Common validators ----------------
  inline Status expect_rowmajor_2d(const Tensor& T, const char* /*name*/) {
    if (!T.is_cuda()) return Status::DeviceMismatch;
    if (!T.is_contiguous_rowmajor_2d()) return Status::LayoutMismatch;
    if (T.desc.dtype != DType::F32 && T.desc.dtype != DType::F16 && T.desc.dtype != DType::BF16)
      return Status::DtypeMismatch;
    if (T.numel() <= 0) return Status::ShapeMismatch;
    return Status::Ok;
  }

  // Bias는 Per-N(1,N) or (N,)만 허용
  inline Status expect_bias_perN_or_null(const Tensor* B, int64_t N) {
    if (!B) return Status::Ok;
    const Tensor& t = *B;
    if (!t.is_cuda()) return Status::DeviceMismatch;
    if (t.desc.shape.size()==1) {
      if (t.desc.shape[0] != N) return Status::ShapeMismatch;
    } else if (t.desc.shape.size()==2) {
      if (!(t.desc.shape[0]==1 && t.desc.shape[1]==N)) return Status::ShapeMismatch;
    } else return Status::ShapeMismatch;
    return Status::Ok;
  }

  } // namespace ai

#else  // BUILD_STANDALONE_OPS

  // 통합 빌드에서는 코어 헤더만 사용 (재선언 금지)
  #include "ai/dispatch.hpp"
  #include "ai/tensor.hpp"
  #include "ai/op_schema.hpp"

#endif // BUILD_STANDALONE_OPS
