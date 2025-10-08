// backends/cuda/ops/rnn/kernels.cu
#include <cuda_runtime.h>
#include <cmath>
#include "backends/cuda/ops/rnn/api.hpp"

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
#endif

namespace ai {

// ========================= device kernels =========================
__global__ void kfill_zero(float* p, int64_t n){
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) p[i] = 0.f;
}

__global__ void kadd_bias_rowwise(float* __restrict__ Y,
                                  const float* __restrict__ b,
                                  int B, int H) {
  const int i = blockIdx.y * blockDim.y + threadIdx.y; // row
  const int j = blockIdx.x * blockDim.x + threadIdx.x; // col
  if (i < B && j < H) {
    Y[static_cast<int64_t>(i) * H + j] += b[j];
  }
}

__global__ void kadd_out(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ C,
                         int64_t n){
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) C[i] = A[i] + B[i];
}

__global__ void kadd_inplace(float* __restrict__ A,
                             const float* __restrict__ B,
                             int64_t n){
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) A[i] += B[i];
}

__global__ void ktanh_out(const float* __restrict__ X,
                          float* __restrict__ Y,
                          int64_t n){
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) Y[i] = tanhf(X[i]);
}

__global__ void ktanh_bwd_from_out(const float* __restrict__ Y,
                                   const float* __restrict__ dY,
                                   float* __restrict__ dZ,
                                   int64_t n){
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const float y = Y[i];
    dZ[i] = dY[i] * (1.f - y * y);
  }
}

// 열(j)마다 하나의 쓰레드가 모든 행(B)을 누적 → 경쟁 없음, atomic 불필요
__global__ void krowwise_sum_accum(const float* __restrict__ M,
                                   float* __restrict__ out,
                                   int B, int H){
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= H) return;
  float acc = 0.f;
  #pragma unroll 4
  for (int i = 0; i < B; ++i) {
    acc += M[static_cast<int64_t>(i) * H + j];
  }
  out[j] += acc; // out이 사전에 0으로 초기화되어 있지 않다면 누적 용도로 사용 가능
}

// in[M,N] (row-major) -> out[N,M] (row-major)
// 타일 기반 전치(공유메모리 사용)로 글로벌 메모리 접근 효율 개선
#ifndef RNN_TPT_TILE
#define RNN_TPT_TILE 32
#endif
#ifndef RNN_TPT_BLOCK_ROWS
#define RNN_TPT_BLOCK_ROWS 8
#endif
__global__ void transpose2d_kernel_tiled(const float* __restrict__ A,
                                         float* __restrict__ AT,
                                         int M, int N) {
  __shared__ float tile[RNN_TPT_TILE][RNN_TPT_TILE + 1]; // bank conflict 회피용 +1

  int x = blockIdx.x * RNN_TPT_TILE + threadIdx.x; // col in A
  int y = blockIdx.y * RNN_TPT_TILE + threadIdx.y; // row in A

  // Load tile from A(y, x) → tile[ty + r, tx]
  for (int r = 0; r < RNN_TPT_TILE; r += RNN_TPT_BLOCK_ROWS) {
    int yy = y + r;
    if (x < N && yy < M) {
      tile[threadIdx.y + r][threadIdx.x] = A[static_cast<int64_t>(yy) * N + x];
    }
  }

  __syncthreads();

  // Write tile^T to AT: (x,y) swapped
  int xt = blockIdx.y * RNN_TPT_TILE + threadIdx.x; // col in AT
  int yt = blockIdx.x * RNN_TPT_TILE + threadIdx.y; // row in AT
  for (int r = 0; r < RNN_TPT_TILE; r += RNN_TPT_BLOCK_ROWS) {
    int yyt = yt + r;
    if (xt < M && yyt < N) {
      AT[static_cast<int64_t>(yyt) * M + xt] = tile[threadIdx.x][threadIdx.y + r];
    }
  }
}

// ========================= host helpers =========================

static inline int div_up_int(int n, int d){ return (n + d - 1) / d; }

static inline int64_t numel_of(const Tensor& t){
  int64_t n = 1;
  for (auto v : t.desc.shape) n *= v;
  return n;
}

static inline float* as_f32(Tensor& t){ return static_cast<float*>(t.data); }
static inline const float* as_cf32(const Tensor& t){ return static_cast<const float*>(t.data); }
static inline cudaStream_t to_cuda(StreamHandle s){ return reinterpret_cast<cudaStream_t>(s); }

// ========================= host wrappers (public) =========================

Status fill_zero(Tensor& t, StreamHandle s){
  const int64_t n = numel_of(t);
  if (n <= 0) return Status::Ok;
  dim3 bs(256), gs(div_up_int(static_cast<int>(n), 256));
  kfill_zero<<<gs, bs, 0, to_cuda(s)>>>(as_f32(t), n);
  return (cudaPeekAtLastError()==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

Status add_bias_rowwise(Tensor& Y, const Tensor& b, int B, int H, StreamHandle s){
  if (B <= 0 || H <= 0) return Status::Ok;
  dim3 bs(32, 8);
  dim3 gs(div_up_int(H, bs.x), div_up_int(B, bs.y));
  kadd_bias_rowwise<<<gs, bs, 0, to_cuda(s)>>>(as_f32(Y), as_cf32(b), B, H);
  return (cudaPeekAtLastError()==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

Status add_out(const Tensor& A, const Tensor& B, Tensor& C, StreamHandle s){
  const int64_t n = numel_of(A);
  if (n <= 0) return Status::Ok;
  dim3 bs(256), gs(div_up_int(static_cast<int>(n), 256));
  kadd_out<<<gs, bs, 0, to_cuda(s)>>>(as_cf32(A), as_cf32(B), as_f32(C), n);
  return (cudaPeekAtLastError()==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

Status add_inplace(Tensor& A, const Tensor& B, StreamHandle s){
  const int64_t n = numel_of(A);
  if (n <= 0) return Status::Ok;
  dim3 bs(256), gs(div_up_int(static_cast<int>(n), 256));
  kadd_inplace<<<gs, bs, 0, to_cuda(s)>>>(as_f32(A), as_cf32(B), n);
  return (cudaPeekAtLastError()==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

Status tanh_out(const Tensor& X, Tensor& Y, StreamHandle s){
  const int64_t n = numel_of(X);
  if (n <= 0) return Status::Ok;
  dim3 bs(256), gs(div_up_int(static_cast<int>(n), 256));
  ktanh_out<<<gs, bs, 0, to_cuda(s)>>>(as_cf32(X), as_f32(Y), n);
  return (cudaPeekAtLastError()==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

Status tanh_bwd_from_out(const Tensor& Y, const Tensor& dY, Tensor& dZ, StreamHandle s){
  const int64_t n = numel_of(Y);
  if (n <= 0) return Status::Ok;
  dim3 bs(256), gs(div_up_int(static_cast<int>(n), 256));
  ktanh_bwd_from_out<<<gs, bs, 0, to_cuda(s)>>>(as_cf32(Y), as_cf32(dY), as_f32(dZ), n);
  return (cudaPeekAtLastError()==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

Status rowwise_sum_accum(const Tensor& M, Tensor& out, int B, int H, StreamHandle s){
  if (H <= 0) return Status::Ok;
  dim3 bs(256), gs(div_up_int(H, 256));
  krowwise_sum_accum<<<gs, bs, 0, to_cuda(s)>>>(as_cf32(M), as_f32(out), B, H);
  return (cudaPeekAtLastError()==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

// in[M,N] -> out[N,M]
Status transpose_2d(const Tensor& A, Tensor& AT, int M, int N, StreamHandle s){
  const float* src = as_cf32(A);
  float* dst       = as_f32(AT);

  // 타일 기반 전치
  dim3 block(RNN_TPT_TILE, RNN_TPT_BLOCK_ROWS);
  dim3 grid(div_up_int(N, RNN_TPT_TILE), div_up_int(M, RNN_TPT_TILE));
  transpose2d_kernel_tiled<<<grid, block, 0, to_cuda(s)>>>(src, dst, M, N);

  return (cudaPeekAtLastError()==cudaSuccess) ? Status::Ok : Status::RuntimeError;
}

} // namespace ai
