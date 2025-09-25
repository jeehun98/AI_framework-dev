#pragma once
#include "ai/tensor.hpp"
#include "ai/dispatch.hpp"
#include <vector>

namespace ai {

// in_desc의 데이터(포인터/디바이스)는 그대로 공유하고
// shape/stride만 permute/expand 결과로 채워 out_desc에 반환합니다.
Status PermuteMakeView(const TensorDesc& in_desc,
                       const std::vector<int>& perm,
                       TensorDesc& out_desc);

Status Transpose2DMakeView(const TensorDesc& in_desc,
                           int dim0, int dim1,
                           TensorDesc& out_desc);

Status ExpandMakeView(const TensorDesc& in_desc,
                      const std::vector<int64_t>& out_shape,
                      TensorDesc& out_desc);

// 선택: out_desc로 바로 Tensor 핸들 조립할 때 쓰는 도우미
inline Tensor MakeViewTensor(void* data, const TensorDesc& desc, Device dev, int device_id=0) {
  return Tensor{data, desc, dev, device_id};
}

} // namespace ai
