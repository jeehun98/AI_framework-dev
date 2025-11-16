// backends/cuda/ops/_common/shim/ai_status.hpp
#pragma once
#include <cstdint>

namespace ai::cuda::shim {

// ---------------- Status ----------------
// 값 영역 예약(ABI 안정): 
//   0: OK
//   1~  99: 공통/일반
// 100~ 129: 형식/배치/디바이스 등 불일치
// 130~ 159: CUDA 관련
// 900~ 999: 기타/미확인
enum class Status : int {
  Ok = 0,

  // 일반
  Invalid          = 1,
  Unimplemented    = 2,
  RuntimeError     = 3,
  InvalidArgument  = 4,

  // 리소스 누락
  MissingInput     = 10,
  MissingOutput    = 11,

  // 불일치/지원 불가
  DeviceMismatch   = 100,
  DtypeMismatch    = 101,
  LayoutMismatch   = 102,
  ShapeMismatch    = 103,
  StrideMismatch   = 104,
  TransposeNotSupported = 105,

  // CUDA 계층
  CUDA_ERROR       = 130,

  Unknown          = 999
};

// ---- ABI anchors (컴파일 타임 가드) ----
static_assert(static_cast<int>(Status::Ok) == 0,           "Status ABI changed");
static_assert(static_cast<int>(Status::CUDA_ERROR) == 130, "Status ABI changed");

// ---- 편의 헬퍼 ----
AI_INLINE constexpr bool status_ok(Status s) {
  return s == Status::Ok;
}

} // namespace ai::cuda::shim

// ---- 매크로: 에러 전파 ----
#ifndef AI_RETURN_IF_ERROR
#define AI_RETURN_IF_ERROR(expr)                                      \
  do {                                                                \
    ::ai::cuda::shim::Status _st__ = (expr);                          \
    if (_st__ != ::ai::cuda::shim::Status::Ok) return _st__;          \
  } while (0)
#endif

// ---- 매크로: CUDA 체크 → Status 변환 ----
// 사용처가 host 런타임(launcher)인 전제. 커널 내부에서는 사용 금지.
#ifndef AI_CUDA_RETURN_IF_ERROR
#define AI_CUDA_RETURN_IF_ERROR(cuda_call)                             \
  do {                                                                 \
    cudaError_t _ce__ = (cuda_call);                                   \
    if (_ce__ != cudaSuccess) {                                        \
      return ::ai::cuda::shim::Status::CUDA_ERROR;                     \
    }                                                                  \
  } while (0)
#endif
