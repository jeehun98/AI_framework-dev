#pragma once
#ifdef AI_USE_NVTX
  #include "nvToolsExt.h"
#endif

namespace ai {
#ifdef AI_USE_NVTX
  struct NvtxRange { explicit NvtxRange(const char* n){ nvtxRangePushA(n); } ~NvtxRange(){ nvtxRangePop(); } };
  #define AI_NVTX_RANGE(name) ::ai::NvtxRange _nvtx_range__(name)
#else
  #define AI_NVTX_RANGE(name) ((void)0)
#endif
} // namespace ai
