#pragma once

#if defined(USE_NVTX)
  #include <nvtx3/nvToolsExt.h>
  #define NVTX_RANGE(name,color) nvtxRangePushEx(&(nvtxEventAttributes_t{0,0,0,color,0,name,0,0}))
  #define NVTX_POP() nvtxRangePop()
#else
  #define NVTX_RANGE(name,color) do{}while(0)
  #define NVTX_POP() do{}while(0)
#endif
