// backends/cuda/ops/_common/shim/workspace.hpp
#pragma once
#include <cstddef>
#include <cstdint>
#include "ai_defs.hpp"  // for AI_INLINE

namespace ai::cuda::shim {

// ------------------------------------------------------------
// Workspace alignment check
// ------------------------------------------------------------
// CUDA Graph / allocator 경로에서 workspace 포인터가 256B 정렬되어 있는지 검사.
// nullptr인 경우(true 반환)는 WS 미사용 상태를 의미.
[[nodiscard]] AI_INLINE bool is_workspace_aligned(const void* p, std::size_t alignment = 256) noexcept {
  if (!p) return true;  // null WS는 무시
  auto addr = reinterpret_cast<std::uintptr_t>(p);
  return (addr % alignment) == 0;
}

} // namespace ai::cuda::shim
