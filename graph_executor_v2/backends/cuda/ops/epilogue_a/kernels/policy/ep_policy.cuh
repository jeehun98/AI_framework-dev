#pragma once
#include <cstdint>

namespace epi {

struct TilePolicy {
  // simple default tile – can be tuned
  static constexpr int TPB = 256;  // threads per block
  static constexpr int VEC = 4;    // vectorize over N if aligned
};

} // namespace epi
