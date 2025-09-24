#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>
#include "backends/cuda/ops/concat/api.hpp"

namespace ai {

// 도우미: s[b:e) 곱
static inline int64_t prod(const std::vector<int64_t>& s, int b, int e){
  int64_t p=1; for (int i=b;i<e;++i) p*=s[i]; return p;
}

Status ConcatCudaLaunch(const std::vector<Tensor>& Xs,
                        Tensor& Y,
                        const ConcatAttrs& a,
                        StreamHandle stream)
{
  const int n = (int)Xs.size();
  if (n<=0) return Status::Invalid;

  const auto& yshape = Y.desc.shape;
  const int nd = (int)yshape.size();
  if (a.axis<0 || a.axis>=nd) return Status::Invalid;

  // 검증: dtype/레이아웃/디바이스/nd 일치 및 non-concat dims 동일
  for (int i=0;i<n;i++){
    const auto& Xi = Xs[i];
    if (Xi.device!=Device::CUDA || Xi.desc.dtype!=DType::F32 || Xi.desc.layout!=Layout::RowMajor)
      return Status::Invalid;
    if ((int)Xi.desc.shape.size()!=nd) return Status::ShapeMismatch;
    for (int d=0; d<nd; ++d){
      if (d==a.axis) continue;
      if (Xi.desc.shape[d]!=yshape[d]) return Status::ShapeMismatch;
    }
  }

  // axis 길이/합, outer/inner, prefix_axis
  std::vector<int64_t> axis_len(n);
  int64_t total_axis=0;
  for (int i=0;i<n;i++){ axis_len[i]=Xs[i].desc.shape[a.axis]; total_axis+=axis_len[i]; }
  if (total_axis!=yshape[a.axis]) return Status::ShapeMismatch;

  const int64_t outer = prod(yshape, 0, a.axis);
  const int64_t inner = prod(yshape, a.axis+1, nd);

  std::vector<int64_t> prefix_axis(n,0);
  for (int i=1;i<n;i++) prefix_axis[i]=prefix_axis[i-1]+axis_len[i-1];

  // 0-size 빠른 종료
  int64_t y_elems=1; for (auto v: yshape) y_elems*=v;
  if (y_elems==0) return Status::Ok;

  auto s = (cudaStream_t)stream;

  // D2D memcpy 루프: 입력 k, 각 outer 슬라이스 o에서 연속 블록 복사
  for (int k=0;k<n;k++){
    const size_t bytes = (size_t)(axis_len[k] * inner) * sizeof(float);
    const float* X = static_cast<const float*>(Xs[k].data);
    float*       Yp = static_cast<float*>(Y.data);

    for (int64_t o=0;o<outer;o++){
      const int64_t src_off = o * (axis_len[k] * inner);
      const int64_t dst_off = o * (total_axis * inner) + (prefix_axis[k] * inner);
      auto err = cudaMemcpyAsync(Yp + dst_off, X + src_off, bytes,
                                 cudaMemcpyDeviceToDevice, s);
      if (err != cudaSuccess) return Status::Invalid;
    }
  }

  // 복사 완료 보장
  auto serr = cudaStreamSynchronize(s);
  return (serr==cudaSuccess) ? Status::Ok : Status::Invalid;
}

} // namespace ai
