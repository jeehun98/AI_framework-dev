#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>
#include "backends/cuda/ops/pad/api.hpp"

namespace ai {

static inline bool is_rowmajor_f32_cuda(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 && t.desc.layout==Layout::RowMajor;
}
static inline cudaStream_t to_cuda(StreamHandle h){ return (cudaStream_t)h; }

// ======== 고정 상한 (필요 시 16 등으로 늘리면 됨) ========
constexpr int PAD_MAXD = 8;

// ---- 커널 메타를 값으로 전달(캡처-세이프) ----
struct PadMeta {
  int   D;
  float value;         // fwd only
  int64_t total_in;    // bwd only
  int64_t total_out;   // fwd only
  int64_t ishape[PAD_MAXD];
  int64_t istride[PAD_MAXD];
  int64_t oshape[PAD_MAXD];
  int64_t ostride[PAD_MAXD];
  int     before[PAD_MAXD];
};

// ---- device helpers ----
__device__ __forceinline__
void linear_to_coords_ll(long long idx, const int64_t* shape, int D, int64_t* coord) {
  #pragma unroll
  for (int i = D - 1; i >= 0; --i) { coord[i] = idx % shape[i]; idx /= shape[i]; }
}

__device__ __forceinline__
int64_t coords_to_offset_ll(const int64_t* coord, const int64_t* stride, int D) {
  int64_t off = 0;
  #pragma unroll
  for (int i=0;i<D;++i) off += coord[i] * stride[i];
  return off;
}

// ---- kernels ----
__global__ void pad_constant_kernel_val(
  const float* __restrict__ x, float* __restrict__ y, PadMeta m)
{
  const long long n = m.total_out;
  for (long long tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < n; tid += 1LL * blockDim.x * gridDim.x)
  {
    int64_t ocoord[PAD_MAXD];
    linear_to_coords_ll(tid, m.oshape, m.D, ocoord);

    bool inside = true;
    int64_t icoord[PAD_MAXD];
    #pragma unroll
    for (int i=0;i<m.D;++i){
      const int64_t xi = ocoord[i] - (int64_t)m.before[i];
      icoord[i] = xi;
      if ((unsigned long long)xi >= (unsigned long long)m.ishape[i]) inside = false;
    }

    const int64_t y_off = coords_to_offset_ll(ocoord, m.ostride, m.D);
    if (!inside) {
      y[y_off] = m.value;
    } else {
      const int64_t x_off = coords_to_offset_ll(icoord, m.istride, m.D);
      y[y_off] = __ldg(x + x_off);
    }
  }
}

__global__ void pad_backward_kernel_val(
  const float* __restrict__ dy, float* __restrict__ dx, PadMeta m)
{
  const long long n = m.total_in;
  for (long long tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < n; tid += 1LL * blockDim.x * gridDim.x)
  {
    int64_t icoord[PAD_MAXD];
    linear_to_coords_ll(tid, m.ishape, m.D, icoord);

    int64_t ocoord[PAD_MAXD];
    #pragma unroll
    for (int i=0;i<m.D;++i) ocoord[i] = icoord[i] + (int64_t)m.before[i];

    const int64_t x_off = coords_to_offset_ll(icoord, m.istride, m.D);
    const int64_t y_off = coords_to_offset_ll(ocoord, m.ostride, m.D);
    dx[x_off] = dy[y_off];
  }
}

// ---- host helpers ----
static inline size_t vec_prod(const std::vector<int64_t>& v){
  size_t p = 1; for (auto x : v) p *= (size_t)x; return p;
}

static inline bool fill_meta_fwd(const Tensor& X, const Tensor& Y,
                                 const PadSpec& s, PadMeta& m)
{
  const int D = (int)X.desc.shape.size();
  if (D <= 0 || D > PAD_MAXD) return false;
  m.D = D;
  m.value = s.value;
  m.total_out = (int64_t)vec_prod(Y.desc.shape);

  for (int i=0;i<D;++i){
    const int64_t xi = X.desc.shape[i];
    const int64_t yi = Y.desc.shape[i];
    const int b = (i < (int)s.before.size()? s.before[i] : 0);
    const int a = (i < (int)s.after.size()?  s.after[i]  : 0);
    if (b < 0 || a < 0) return false;
    if (yi != xi + b + a) return false;
    m.ishape[i]  = xi;
    m.istride[i] = X.desc.stride[i];
    m.oshape[i]  = yi;
    m.ostride[i] = Y.desc.stride[i];
    m.before[i]  = b;
  }
  // 나머지 채널 클리어
  for (int i=D;i<PAD_MAXD;++i){
    m.ishape[i]=m.oshape[i]=1;
    m.istride[i]=m.ostride[i]=0;
    m.before[i]=0;
  }
  return true;
}

static inline bool fill_meta_bwd(const Tensor& dY, const Tensor& dX,
                                 const PadSpec& s, PadMeta& m)
{
  const int D = (int)dX.desc.shape.size();
  if (D <= 0 || D > PAD_MAXD) return false;
  m.D = D;
  m.total_in = (int64_t)vec_prod(dX.desc.shape);

  for (int i=0;i<D;++i){
    const int64_t xi = dX.desc.shape[i];
    const int64_t yi = dY.desc.shape[i];
    const int b = (i < (int)s.before.size()? s.before[i] : 0);
    const int a = (i < (int)s.after.size()?  s.after[i]  : 0);
    if (b < 0 || a < 0) return false;
    if (yi != xi + b + a) return false;
    m.ishape[i]  = xi;                  // in = dX
    m.istride[i] = dX.desc.stride[i];
    m.oshape[i]  = yi;                  // out = dY
    m.ostride[i] = dY.desc.stride[i];
    m.before[i]  = b;
  }
  for (int i=D;i<PAD_MAXD;++i){
    m.ishape[i]=m.oshape[i]=1;
    m.istride[i]=m.ostride[i]=0;
    m.before[i]=0;
  }
  return true;
}

// ---- launchers (capture-safe) ----
// 기존 4-인자 정의 → 삭제 또는 수정
// Status PadCudaLaunch(const Tensor& X, Tensor& Y, const PadSpec& s, StreamHandle stream)
// Status PadBackwardCudaLaunch(const Tensor& dY, Tensor& dX, const PadSpec& s, StreamHandle stream)

// === 수정된 5-인자 정의 ===
Status PadCudaLaunch(const Tensor& X, Tensor& Y,
                     const PadSpec& s,
                     StreamHandle stream,
                     const PadWorkspaceFwd* /*ws_fwd*/)   // 사용 안 해도 시그니처만 맞추면 OK
{
  if (!is_rowmajor_f32_cuda(X) || !is_rowmajor_f32_cuda(Y)) return Status::Invalid;
  if (X.desc.shape.size() != Y.desc.shape.size()) return Status::ShapeMismatch;

  PadMeta m{};
  if (!fill_meta_fwd(X, Y, s, m)) return Status::Invalid;

  const long long n = m.total_out;
  int block = 256;
  int grid  = (int)std::min( (n + block - 1) / block, 1LL * 65535 );

  pad_constant_kernel_val<<<grid, block, 0, to_cuda(stream)>>>(
      static_cast<const float*>(X.data),
      static_cast<float*>(Y.data),
      m
  );
  auto e = cudaPeekAtLastError();
  return (e==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

Status PadBackwardCudaLaunch(const Tensor& dY, Tensor& dX,
                             const PadSpec& s,
                             StreamHandle stream,
                             const PadWorkspaceBwd* /*ws_bwd*/)
{
  if (!is_rowmajor_f32_cuda(dY) || !is_rowmajor_f32_cuda(dX)) return Status::Invalid;
  if (dX.desc.shape.size() != dY.desc.shape.size()) return Status::ShapeMismatch;

  PadMeta m{};
  if (!fill_meta_bwd(dY, dX, s, m)) return Status::Invalid;

  const long long n = m.total_in;
  int block = 256;
  int grid  = (int)std::min( (n + block - 1) / block, 1LL * 65535 );

  pad_backward_kernel_val<<<grid, block, 0, to_cuda(stream)>>>(
      static_cast<const float*>(dY.data),
      static_cast<float*>(dX.data),
      m
  );
  auto e = cudaPeekAtLastError();
  return (e==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}


} // namespace ai
