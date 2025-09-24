#pragma once
#include "ai/tensor.hpp"
#include "ai/dispatch.hpp"
#include <vector>

namespace ai {

enum class ReduceOp : int { Sum=0, Mean=1, Max=2, Min=3 };

struct ReduceAttrs {
  // 줄일 축들 (음수 가능: 파이썬처럼 음수 축 허용)
  std::vector<int> axes;
  bool keepdim{false};
  ReduceOp op{ReduceOp::Sum};
};

// 🔑 이제 PackMKN 정의를 api.hpp로 이동
struct PackMKN {
  int64_t M{1}, K{1}, N{1};
  int64_t sM{0}, sK{0}, sN{0}; // 요소 단위 stride
};

Status ReduceCudaLaunch(const Tensor& X, Tensor& Y,
                        const ReduceAttrs& attrs, StreamHandle stream);

} // namespace ai
