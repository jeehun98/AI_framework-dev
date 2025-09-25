#include "ai/tensor.hpp"
#include "ai/dispatch.hpp"
#include "backends/cuda/ops/memory/api.hpp"

namespace ai { namespace ops {

// ---- 공용: RowMajor contiguous stride 생성 ----
static inline std::vector<int64_t> make_contig_stride(const std::vector<int64_t>& shape){
  const int D = (int)shape.size();
  std::vector<int64_t> s(D,1);
  for (int i=D-2;i>=0;--i) s[i] = s[i+1]*shape[i+1];
  return s;
}

// ---- View: permute ----
int permute_view(const Tensor& X, TensorDesc& out_desc, const std::vector<int>& axes)
{
  const int D = (int)X.desc.shape.size();
  if ((int)axes.size()!=D) return -2;

  // 유효성 (0..D-1, unique)
  std::vector<int> seen(D,0);
  for (int a: axes){
    if (a<0 || a>=D) return -2;
    if (seen[a]) return -2;
    seen[a]=1;
  }
  out_desc = X.desc;
  out_desc.shape.resize(D);
  out_desc.stride.resize(D);
  for (int i=0;i<D;++i){
    out_desc.shape[i]  = X.desc.shape[axes[i]];
    out_desc.stride[i] = X.desc.stride[axes[i]];
  }
  return 0;
}

// ---- View: expand (브로드캐스트) ----
// 규칙: in.shape[d] == out.shape[d] 이면 stride 유지,
//      in.shape[d] == 1 이면 stride=0 (broadcast)
// 그 외는 오류
int expand_view(const Tensor& X, TensorDesc& out_desc, const std::vector<int64_t>& out_shape)
{
  const int Dout = (int)out_shape.size();
  const int Din  = (int)X.desc.shape.size();
  // 좌측 pad 개념으로 Din을 Dout에 맞춤
  if (Din > Dout) return -2;

  out_desc = X.desc;
  out_desc.shape  = out_shape;
  out_desc.stride = std::vector<int64_t>(Dout, 0);

  // 왼쪽 채우기(= 앞쪽 축 추가)
  int pad = Dout - Din;
  for (int d=0; d<Dout; ++d){
    int src = d - pad; // <0 이면 새로 생긴 leading dim
    if (src < 0){
      // 새로 생긴 leading dim은 길이가 out_shape[d].
      // broadcast 규칙상 입력 길이는 1로 간주 → stride=0
      out_desc.stride[d] = 0;
    } else {
      int64_t in_len  = X.desc.shape[src];
      int64_t out_len = out_shape[d];
      if (in_len == out_len){
        out_desc.stride[d] = X.desc.stride[src];
      } else if (in_len == 1){
        out_desc.stride[d] = 0; // broadcast
      } else {
        return -3; // 호환 불가
      }
    }
  }
  return 0;
}

// ---- Run: contiguous copy (CUDA 백엔드 호출) ----
int contiguous_copy_run(const Tensor& X, Tensor& Y, StreamHandle stream)
{
  // Y는 반드시 RowMajor contiguous stride
  if (X.desc.dtype != DType::F32 || Y.desc.dtype != DType::F32) return -2;
  if (X.desc.shape != Y.desc.shape) return -3;

  if (Y.device == Device::CUDA && X.device == Device::CUDA){
    auto st = ai::ContiguousCopyCudaLaunch(X, Y, stream);
    return (st==Status::Ok) ? 0 : -7;
  }
  // (필요시) CPU path 추가 가능
  return -9; // Not implemented
}

}} // namespace ai::ops
