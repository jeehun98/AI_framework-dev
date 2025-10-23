#pragma once
#include <cstdint>

#if defined(USE_NVTX)
  #include <nvToolsExt.h>
#endif

namespace ai { namespace nvtx {

#if defined(USE_NVTX)

// 색 팔레트 (타임라인에서 보이는 ARGB)
enum class Color : uint32_t {
  Red      = 0xFFFF4C4C,
  Green    = 0xFF66CC66,
  Blue     = 0xFF4C7DFF,
  Orange   = 0xFFFFAA66,
  Purple   = 0xFFB266FF,
  Yellow   = 0xFFFFD24C,
  Teal     = 0xFF4CCCCC,
  Gray     = 0xFF808080,
  Cyan     = 0xFF66CCFF,
  Magenta  = 0xFFFF66FF,
};

// RAII range
struct Range {
  Range(const char* name, uint32_t argb_color) {
    nvtxEventAttributes_t attr{};
    attr.version     = NVTX_VERSION;
    attr.size        = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType   = NVTX_COLOR_ARGB;
    attr.color       = argb_color;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    nvtxRangePushEx(&attr);
  }
  ~Range() { nvtxRangePop(); }
};

inline void mark(const char* name) { nvtxMarkA(name); }

#else // !USE_NVTX

enum class Color : uint32_t {
  Red=0,Green=0,Blue=0,Orange=0,Purple=0,Yellow=0,Teal=0,Gray=0,Cyan=0,Magenta=0
};
struct Range { Range(const char*, uint32_t) {} };
inline void mark(const char*) {}

#endif // USE_NVTX

}} // namespace ai::nvtx

// ---- 유니크 변수명 도우미 ----
#define AI_NVTX_CONCAT_IMPL(x,y) x##y
#define AI_NVTX_CONCAT(x,y)      AI_NVTX_CONCAT_IMPL(x,y)
#if defined(__COUNTER__)
  #define AI_NVTX_UNIQUE(base)   AI_NVTX_CONCAT(base, __COUNTER__)
#else
  #define AI_NVTX_UNIQUE(base)   AI_NVTX_CONCAT(base, __LINE__)
#endif

// ---- 매크로(프로젝트 전역) ----
#define NVTX_RANGE(name, color_enum) \
  ::ai::nvtx::Range AI_NVTX_UNIQUE(_nvtx_range_) { (name), static_cast<uint32_t>(color_enum) }

#define NVTX_MARK(name) ::ai::nvtx::mark(name)
