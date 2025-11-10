#pragma once
#include <cuda_runtime_api.h>

namespace ai {
// ---------------- Stream ----------------
using StreamHandle = cudaStream_t;              // 타입 세이프
inline cudaStream_t as_cuda_stream(StreamHandle s){ return s; }
} // namespace ai
