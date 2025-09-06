#pragma once
// Define REGEMM_USE_NVTX=1 to enable real NVTX.
// Otherwise provide no-op shims.
#if defined(REGEMM_USE_NVTX) && REGEMM_USE_NVTX
  #include <nvToolsExt.h>
  struct NvtxRangeScope {
    explicit NvtxRangeScope(const char* name){ nvtxRangePushA(name); }
    ~NvtxRangeScope(){ nvtxRangePop(); }
  };
#else
  inline void nvtxRangePushA(const char*) {}
  inline void nvtxRangePop() {}
  struct NvtxRangeScope { explicit NvtxRangeScope(const char*) {} };
#endif
