#pragma once
#include "ai/tensor.hpp"
#include "ai/dispatch.hpp"

namespace ai {

// X: float32 contiguous, Index: int32 contiguous, Y: float32 contiguous
// axis 기준 인덱싱. Y.shape == Index.shape, 그리고
//   * Y.shape[d] == X.shape[d] for d != axis
//   * Index.shape[axis] == (출력축 길이 M), 값 범위: [0, X.shape[axis])
Status GatherCudaLaunch(const Tensor& X, const Tensor& Index, Tensor& Y,
                        int axis, StreamHandle stream);

// Out[o, Index[o,m,i], i] += Src[o,m,i] (axis 기준)
// Out, Src: float32 contiguous. Index: int32 contiguous
// Out.shape: ...K...  (축 길이 K)
// Src.shape/Index.shape: 동일, 단 축 길이 M (scatter되는 길이).
Status ScatterAddCudaLaunch(Tensor& Out, const Tensor& Index, const Tensor& Src,
                            int axis, StreamHandle stream);

} // namespace ai
