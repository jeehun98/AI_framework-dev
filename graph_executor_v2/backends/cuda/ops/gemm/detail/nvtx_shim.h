#pragma once
// GEMM 쪽 detail 에서 쓰던 shim — 공용 nvtx.hpp 를 그대로 사용
#include "../../_common/shim/nvtx.hpp"
using NVTX_COLOR = ::ai::nvtx::Color;  // 기존 코드 호환: NVTX_COLOR::Gray 등
