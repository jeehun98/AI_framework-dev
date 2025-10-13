// backends/cuda/ops/pool2d/launcher.cu
#include <cuda_runtime.h>
#include "backends/cuda/ops/pool2d/api.hpp"

namespace ai {

static inline bool is_nchw_f32_4d_cuda(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor && t.desc.shape.size()==4;
}
static inline bool is_nchw_i32_4d_cuda(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::I32 &&
         t.desc.layout==Layout::RowMajor && t.desc.shape.size()==4;
}

static inline cudaStream_t to_cuda(StreamHandle h){ return (cudaStream_t)h; }

// ===== 커널 런처 (kernels.cu 와 정확히 동일 시그니처) =====
void maxpool2d_fwd_kernel_launcher(const float*, float*, int32_t*,
                                   int,int,int,int,int,int,int,int,int,int,int,int,bool,
                                   cudaStream_t);
void maxpool2d_bwd_kernel_launcher(const float*, const int32_t*, float*,
                                   int,int,int,int,int,int,int,int,int,int,int,int,bool,
                                   cudaStream_t);
void avgpool2d_fwd_kernel_launcher(const float*, float*,
                                   int,int,int,int,int,int,int,int,int,int,int,int,bool,bool,
                                   cudaStream_t);
void avgpool2d_bwd_kernel_launcher(const float*, float*,
                                   int,int,int,int,int,int,int,int,int,int,int,int,bool,bool,
                                   cudaStream_t);

// ===== 출력 크기 공식(커널과 동일) =====
static inline int div_up_host(int a, int b){ return (a + b - 1) / b; }
static inline void out_dims_host(
  int H,int W,int kH,int kW,int sH,int sW,int pH,int pW,int dH,int dW,bool ceil_mode,
  int& Ho,int& Wo)
{
  const int effKH = (kH - 1) * dH + 1;
  const int effKW = (kW - 1) * dW + 1;
  const int aH = H + 2 * pH - effKH;
  const int aW = W + 2 * pW - effKW;
  if (ceil_mode) {
    Ho = (aH >= 0 ? div_up_host(aH, sH) + 1 : 0);
    Wo = (aW >= 0 ? div_up_host(aW, sW) + 1 : 0);
  } else {
    Ho = (aH >= 0 ? (aH / sH) + 1 : 0);
    Wo = (aW >= 0 ? (aW / sW) + 1 : 0);
  }
  if (Ho < 0) Ho = 0;
  if (Wo < 0) Wo = 0;
}

// ================= MaxPool2D =================
Status MaxPool2DCudaLaunch(const Tensor& X, Tensor& Y, Tensor* Indices,
                           const Pool2DAttrs& a, StreamHandle stream,
                           const MaxPool2DWorkspaceFwd* ws_fwd /*=nullptr*/)
{
  if (!is_nchw_f32_4d_cuda(X) || !is_nchw_f32_4d_cuda(Y)) return Status::Invalid;
  if (X.desc.shape[0]!=Y.desc.shape[0] || X.desc.shape[1]!=Y.desc.shape[1]) return Status::ShapeMismatch;

  // Y가 기대하는 (Ho,Wo)인지 확인
  {
    int Ho=0, Wo=0;
    out_dims_host((int)X.desc.shape[2], (int)X.desc.shape[3],
                  a.kH,a.kW,a.sH,a.sW,a.pH,a.pW,a.dH,a.dW,a.ceil_mode, Ho, Wo);
    if ((int)Y.desc.shape[2]!=Ho || (int)Y.desc.shape[3]!=Wo) return Status::ShapeMismatch;
  }

  // 인덱스 목적지 선택: 우선순위 = 명시적 Tensor(Indices) > WS(indices) > nullptr
  int32_t* ind = nullptr;
  if (Indices){
    if (!is_nchw_i32_4d_cuda(*Indices)) return Status::Invalid;
    if (Indices->desc.shape != Y.desc.shape) return Status::ShapeMismatch;
    ind = static_cast<int32_t*>(Indices->data);
  } else if (ws_fwd && ws_fwd->indices){
    // ws_fwd->indices는 [N,C,Ho,Wo] 크기로 호출자(그래프 아레나)에서 올바르게 할당되어야 함
    ind = ws_fwd->indices; // 모양 검증 불가: 호출자 책임
  }

  maxpool2d_fwd_kernel_launcher(
    static_cast<const float*>(X.data),
    static_cast<float*>(Y.data),
    ind,
    (int)X.desc.shape[0], (int)X.desc.shape[1], (int)X.desc.shape[2], (int)X.desc.shape[3],
    a.kH,a.kW,a.sH,a.sW,a.pH,a.pW,a.dH,a.dW,a.ceil_mode,
    to_cuda(stream)
  );

  cudaError_t e = cudaPeekAtLastError();
  if (e != cudaSuccess) return Status::RuntimeError;
  return Status::Ok;
}

Status MaxPool2DBackwardCudaLaunch(const Tensor& dY, Tensor& dX,
                                   const Tensor* Indices,            // nullable
                                   const Pool2DAttrs& a, StreamHandle stream,
                                   const MaxPool2DWorkspaceBwd* ws_bwd /*=nullptr*/)
{
  if (!is_nchw_f32_4d_cuda(dY) || !is_nchw_f32_4d_cuda(dX))
    return Status::Invalid;

  // dX: (N,C,H,W), dY: (N,C,Ho,Wo)
  const int N  = (int)dX.desc.shape[0];
  const int C  = (int)dX.desc.shape[1];
  const int H  = (int)dX.desc.shape[2];
  const int W  = (int)dX.desc.shape[3];

  if ((int)dY.desc.shape[0]!=N || (int)dY.desc.shape[1]!=C) return Status::ShapeMismatch;

  // Ho,Wo 일치 검증
  int Ho=0, Wo=0;
  out_dims_host(H, W, a.kH,a.kW,a.sH,a.sW,a.pH,a.pW,a.dH,a.dW,a.ceil_mode, Ho, Wo);
  if ((int)dY.desc.shape[2]!=Ho || (int)dY.desc.shape[3]!=Wo) return Status::ShapeMismatch;

  // 인덱스 소스 선택: 명시적 Tensor > WS(indices). 둘 다 없으면 에러(현재 커널이 재계산 미지원).
  const int32_t* ind = nullptr;
  if (Indices){
    if (!is_nchw_i32_4d_cuda(*Indices)) return Status::Invalid;
    if (Indices->desc.shape != dY.desc.shape) return Status::ShapeMismatch;
    ind = static_cast<const int32_t*>(Indices->data);
  } else if (ws_bwd && ws_bwd->indices){
    ind = ws_bwd->indices; // 크기 책임은 호출자
  } else {
    // 재계산 커널이 없다면 인덱스는 필수
    return Status::Invalid;
  }

  // dX = 0 초기화 (atomicAdd 누적)
  cudaError_t e = cudaMemsetAsync(dX.data, 0,
    sizeof(float) * (size_t)N*C*H*W, to_cuda(stream));
  if (e != cudaSuccess) return Status::Invalid;

  maxpool2d_bwd_kernel_launcher(
    static_cast<const float*>(dY.data),
    ind,
    static_cast<float*>(dX.data),
    N, C, H, W,
    a.kH,a.kW,a.sH,a.sW,a.pH,a.pW,a.dH,a.dW,a.ceil_mode,
    to_cuda(stream)
  );

  e = cudaPeekAtLastError();
  if (e != cudaSuccess) return Status::RuntimeError;
  return Status::Ok;
}

// ================= AvgPool2D =================
Status AvgPool2DCudaLaunch(const Tensor& X, Tensor& Y,
                           const Pool2DAttrs& a, StreamHandle stream,
                           const AvgPool2DWorkspaceFwd* /*ws_fwd*/ /*=nullptr*/)
{
  if (!is_nchw_f32_4d_cuda(X) || !is_nchw_f32_4d_cuda(Y)) return Status::Invalid;
  if (X.desc.shape[0]!=Y.desc.shape[0] || X.desc.shape[1]!=Y.desc.shape[1]) return Status::ShapeMismatch;

  // Y 모양 검증
  {
    int Ho=0, Wo=0;
    out_dims_host((int)X.desc.shape[2], (int)X.desc.shape[3],
                  a.kH,a.kW,a.sH,a.sW,a.pH,a.pW,a.dH,a.dW,a.ceil_mode, Ho, Wo);
    if ((int)Y.desc.shape[2]!=Ho || (int)Y.desc.shape[3]!=Wo) return Status::ShapeMismatch;
  }

  avgpool2d_fwd_kernel_launcher(
    static_cast<const float*>(X.data),
    static_cast<float*>(Y.data),
    (int)X.desc.shape[0], (int)X.desc.shape[1], (int)X.desc.shape[2], (int)X.desc.shape[3],
    a.kH,a.kW,a.sH,a.sW,a.pH,a.pW,a.dH,a.dW,a.ceil_mode, a.count_include_pad,
    to_cuda(stream)
  );
  cudaError_t e = cudaPeekAtLastError();
  if (e != cudaSuccess) return Status::RuntimeError;
  return Status::Ok;
}

Status AvgPool2DBackwardCudaLaunch(const Tensor& dY, Tensor& dX,
                                   const Pool2DAttrs& a, StreamHandle stream,
                                   const AvgPool2DWorkspaceBwd* /*ws_bwd*/ /*=nullptr*/)
{
  if (!is_nchw_f32_4d_cuda(dY) || !is_nchw_f32_4d_cuda(dX)) return Status::Invalid;

  // (N,C)만 일치하면 됨
  if ((int)dY.desc.shape[0]!=(int)dX.desc.shape[0] ||
      (int)dY.desc.shape[1]!=(int)dX.desc.shape[1]) return Status::ShapeMismatch;

  // 기대 dY(Ho,Wo) 검증
  {
    int Ho=0, Wo=0;
    out_dims_host((int)dX.desc.shape[2], (int)dX.desc.shape[3],
                  a.kH,a.kW,a.sH,a.sW,a.pH,a.pW,a.dH,a.dW,a.ceil_mode, Ho, Wo);
    if ((int)dY.desc.shape[2]!=Ho || (int)dY.desc.shape[3]!=Wo) return Status::ShapeMismatch;
  }

  // dX = 0 초기화 (atomicAdd)
  cudaError_t e = cudaMemsetAsync(dX.data, 0,
    sizeof(float) * (size_t)dX.desc.shape[0]*dX.desc.shape[1]*dX.desc.shape[2]*dX.desc.shape[3],
    to_cuda(stream));
  if (e != cudaSuccess) return Status::Invalid;

  avgpool2d_bwd_kernel_launcher(
    static_cast<const float*>(dY.data),
    static_cast<float*>(dX.data),
    (int)dX.desc.shape[0], (int)dX.desc.shape[1], (int)dX.desc.shape[2], (int)dX.desc.shape[3],
    a.kH,a.kW,a.sH,a.sW,a.pH,a.pW,a.dH,a.dW,a.ceil_mode, a.count_include_pad,
    to_cuda(stream)
  );
  e = cudaPeekAtLastError();
  if (e != cudaSuccess) return Status::RuntimeError;
  return Status::Ok;
}

} // namespace ai
