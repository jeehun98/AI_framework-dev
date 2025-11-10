// backends/cuda/ops/_common/shim/ai_nvtx.hpp
#pragma once
#include <cstdint>

// ==== 토글 플래그 통일 ====
// 과거 AI_USE_NVTX / USE_NVTX 둘 다 지원. 새 표준은 AI_ENABLE_NVTX.
#if defined(AI_USE_NVTX) || defined(USE_NVTX)
  #ifndef AI_ENABLE_NVTX
    #define AI_ENABLE_NVTX
  #endif
#endif

#if defined(AI_ENABLE_NVTX)
  #include <nvToolsExt.h>
#endif

namespace ai { namespace nvtx {

// ==== 색상 팔레트 (ARGB) ====
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

#if defined(AI_ENABLE_NVTX)

// ==== NVTX 활성 ====
struct Range {
  // 이름만: 기본 Gray
  explicit Range(const char* name)
    : Range(name, static_cast<uint32_t>(Color::Gray)) {}

  // 이름 + ARGB
  Range(const char* name, uint32_t argb_color) {
    nvtxEventAttributes_t attr{};
    attr.version       = NVTX_VERSION;
    attr.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType     = NVTX_COLOR_ARGB;
    attr.color         = argb_color;
    attr.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    nvtxRangePushEx(&attr);
  }
  ~Range() { nvtxRangePop(); }
};

inline void mark(const char* name) { nvtxMarkA(name); }

#else
// ==== NVTX 비활성: 전부 no-op ====
struct Range {
  explicit Range(const char*) {}
  Range(const char*, uint32_t) {}
  ~Range() {}
};
inline void mark(const char*) {}

#endif // AI_ENABLE_NVTX

}} // namespace ai::nvtx

// ==== 유니크 변수명 도우미 ====
#define AI_NVTX_CONCAT_IMPL(x,y) x##y
#define AI_NVTX_CONCAT(x,y)      AI_NVTX_CONCAT_IMPL(x,y)
#if defined(__COUNTER__)
  #define AI_NVTX_UNIQUE(base)   AI_NVTX_CONCAT(base, __COUNTER__)
#else
  #define AI_NVTX_UNIQUE(base)   AI_NVTX_CONCAT(base, __LINE__)
#endif

// ==== 프로젝트 표준 매크로 ====
// 색상 지정
#define AI_NVTX_RANGE(name, color_enum) \
  ::ai::nvtx::Range AI_NVTX_UNIQUE(_nvtx_range_) { (name), static_cast<uint32_t>(color_enum) }

// 기본색(Gray)
#define AI_NVTX_RANGE_GRAY(name) \
  ::ai::nvtx::Range AI_NVTX_UNIQUE(_nvtx_range_) { (name) }

// 포인트 마킹
#define AI_NVTX_MARK(name) ::ai::nvtx::mark(name)

// ==== 과거 호환 별칭 ====
// 예전 코드: NVTX_RANGE("tag", ai::nvtx::Color::Blue);
#ifndef NVTX_RANGE
  #define NVTX_RANGE(name, color_enum) AI_NVTX_RANGE(name, color_enum)
#endif
#ifndef NVTX_MARK
  #define NVTX_MARK(name) AI_NVTX_MARK(name)
#endif
// 예전 코드: AI_NVTX_RANGE("tag") 스타일을 썼다면 아래로 대체
#ifndef AI_NVTX_RANGE_NO_COLOR
  #define AI_NVTX_RANGE_NO_COLOR(name) AI_NVTX_RANGE_GRAY(name)
#endif
