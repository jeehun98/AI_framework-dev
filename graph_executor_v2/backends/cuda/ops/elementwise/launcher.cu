#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include "backends/cuda/ops/elementwise/api.hpp"

namespace ai {

static inline bool is_f32_cuda_rowmajor(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 && t.desc.layout==Layout::RowMajor;
}
static inline cudaStream_t to_cuda(StreamHandle h){ return (cudaStream_t)h; }

// kernels (정의는 kernels.cu)
void ewise_unary_kernel_launcher(const float* X, float* Y,
                                 const int64_t* y_shape,
                                 const int64_t* a_stride,
                                 int nd, int64_t nElem,
                                 int op, float alpha, float cmin, float cmax, float eps,
                                 cudaStream_t s);
void ewise_binary_kernel_launcher(const float* A, const float* B, float* Y,
                                  const int64_t* y_shape,
                                  const int64_t* a_stride,
                                  const int64_t* b_stride,
                                  int nd, int64_t nElem,
                                  int op, float alpha, float beta, float eps,
                                  cudaStream_t s);

// 공용: 브로드캐스트 stride 계산 (RowMajor 가정)
static void compute_broadcast_strides(const std::vector<int64_t>& out_shape,
                                      const std::vector<int64_t>& in_shape,
                                      int64_t* out_stride /*len=nd*/)
{
  const int nd = (int)out_shape.size();
  // 입력 stride 계산(연속 가정)
  std::vector<int64_t> base_stride(nd, 0);
  {
    int64_t s = 1;
    for (int i=(int)in_shape.size()-1; i>=0; --i){
      base_stride[i + (nd - (int)in_shape.size())] = s;
      s *= in_shape[i];
    }
  }
  // 브로드캐스트: out_dim != in_dim(==1) 이면 stride=0
  int shift = nd - (int)in_shape.size();
  for (int i=0;i<nd;i++){
    int64_t in_dim = (i<shift) ? 1 : in_shape[i-shift];
    out_stride[i] = (in_dim==1) ? 0 : base_stride[i];
  }
}

static int64_t numel(const std::vector<int64_t>& shape){
  int64_t n=1; for (auto v: shape) n*=v; return n;
}

Status EWiseUnaryCudaLaunch(const Tensor& X, Tensor& Y,
                            UnaryOp op, const EWiseUnaryAttrs& attrs,
                            StreamHandle stream)
{
  if (!is_f32_cuda_rowmajor(X) || !is_f32_cuda_rowmajor(Y)) return Status::Invalid;
  if (X.desc.shape.empty() || Y.desc.shape.empty()) return Status::Invalid;

  // Y shape가 최종(out)이며, X는 브로드캐스트 가능해야 함
  // out_nd는 Y.nd
  const int nd = (int)Y.desc.shape.size();
  if (nd > 8) return Status::Invalid;

  // X가 Y로 브로드캐스트 가능한지 검사
  // 뒤에서부터 매칭: in_dim==out_dim or in_dim==1
  {
    int ix = (int)X.desc.shape.size()-1;
    for (int oy = nd-1; oy>=0; --oy){
      int64_t out_d = Y.desc.shape[oy];
      int64_t in_d  = (ix>=0)? X.desc.shape[ix] : 1;
      if (!(in_d==out_d || in_d==1)) return Status::ShapeMismatch;
      --ix;
    }
  }

  // stride/shape 패킹
  int64_t y_shape[8], a_stride[8];
  for (int i=0;i<nd;i++) y_shape[i]=Y.desc.shape[i];

  compute_broadcast_strides(Y.desc.shape, X.desc.shape, a_stride);

  const int64_t nElem = numel(Y.desc.shape);
  ewise_unary_kernel_launcher(
    static_cast<const float*>(X.data),
    static_cast<float*>(Y.data),
    y_shape, a_stride, nd, nElem,
    (int)op, attrs.alpha, attrs.clip_min, attrs.clip_max, attrs.eps,
    to_cuda(stream)
  );
  return (cudaPeekAtLastError()==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

Status EWiseBinaryCudaLaunch(const Tensor& A, const Tensor& B, Tensor& Y,
                             BinaryOp op, const EWiseBinaryAttrs& attrs,
                             StreamHandle stream)
{
  if (!is_f32_cuda_rowmajor(A) || !is_f32_cuda_rowmajor(B) || !is_f32_cuda_rowmajor(Y))
    return Status::Invalid;

  const int nd = (int)Y.desc.shape.size();
  if (nd>8) return Status::Invalid;

  auto can_bc = [&](const std::vector<int64_t>& in){
    int ii = (int)in.size()-1;
    for (int oy = nd-1; oy>=0; --oy){
      int64_t out_d = Y.desc.shape[oy];
      int64_t in_d  = (ii>=0)? in[ii] : 1;
      if (!(in_d==out_d || in_d==1)) return false;
      --ii;
    }
    return true;
  };
  if (!can_bc(A.desc.shape) || !can_bc(B.desc.shape)) return Status::ShapeMismatch;

  int64_t y_shape[8], a_stride[8], b_stride[8];
  for (int i=0;i<nd;i++) y_shape[i]=Y.desc.shape[i];

  compute_broadcast_strides(Y.desc.shape, A.desc.shape, a_stride);
  compute_broadcast_strides(Y.desc.shape, B.desc.shape, b_stride);

  const int64_t nElem = numel(Y.desc.shape);
  ewise_binary_kernel_launcher(
    static_cast<const float*>(A.data),
    static_cast<const float*>(B.data),
    static_cast<float*>(Y.data),
    y_shape, a_stride, b_stride, nd, nElem,
    (int)op, attrs.alpha, attrs.beta, attrs.eps,
    to_cuda(stream)
  );
  return (cudaPeekAtLastError()==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

} // namespace ai
