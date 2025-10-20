#pragma once
#include <cuda_runtime.h>
#include "ep_policy.cuh"

namespace epi {

inline dim3 compute_grid(int64_t M, int64_t N) {
  int64_t elems = M * N;
  int blocks = static_cast<int>((elems + TilePolicy::TPB - 1) / TilePolicy::TPB);
  return dim3(blocks, 1, 1);
}
inline dim3 compute_block() { return dim3(TilePolicy::TPB, 1, 1); }

} // namespace epi
