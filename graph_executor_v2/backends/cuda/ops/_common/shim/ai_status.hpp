// backends/cuda/ops/_common/shim/ai_status.hpp

#pragma once
#include <cstdint>

namespace ai {

// ---------------- Status ----------------
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
  InvalidArgument = 106,

  MissingInput   = 110,
  MissingOutput  = 111,

  CUDA_ERROR = 130,
  Unknown = 999
};

#ifndef AI_RETURN_IF_ERROR
#define AI_RETURN_IF_ERROR(expr) do {                      \
  ::ai::Status _st__ = (expr);                             \
  if (_st__ != ::ai::Status::Ok) return _st__;             \
} while(0)
#endif

} // namespace ai
