// _common/shim/workspace.hpp
#pragma once
#include <cstddef>
#include <cstdint>
namespace ai::cuda::shim {
inline bool is_workspace_aligned(const void* p, std::size_t alignment=256){
  return (reinterpret_cast<std::uintptr_t>(p)%alignment)==0;
}
} // ns
