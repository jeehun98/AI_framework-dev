// backends/cuda/ops/rnn/launcher.cu
#include <cuda_runtime.h>
#include <cstring>
#include "backends/cuda/ops/rnn/api.hpp"

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/op_schema.hpp"
#endif

#ifndef AI_RETURN_IF_ERROR
#define AI_RETURN_IF_ERROR(expr) do { ::ai::Status _st__=(expr); if(_st__!=::ai::Status::Ok) return _st__; } while(0)
#endif
#ifndef AI_CUDA_TRY
#define AI_CUDA_TRY(expr) do { cudaError_t _e__=(expr); if(_e__!=cudaSuccess) return ::ai::Status::RuntimeError; } while(0)
#endif

namespace ai {

static inline Tensor make_view2d(void* p, int64_t rows, int64_t cols){
  Tensor t;
  t.data         = p;
  t.device       = Device::CUDA;
  t.device_index = 0;
  t.desc.dtype   = DType::F32;
  t.desc.layout  = Layout::RowMajor;
  t.desc.shape   = {rows, cols};
  t.desc.stride  = {cols, 1};
  return t;
}

static inline Tensor row_slice_2d(const Tensor& base, int64_t row0, int64_t rows){
  const int64_t N = base.desc.shape[1];
  auto* p = static_cast<char*>(base.data) + (row0 * N) * sizeof(float);
  return make_view2d(p, rows, N);
}

static inline int div_up(int n, int d){ return (n + d - 1) / d; }

#ifdef BUILD_STANDALONE_OPS
// ------------------------------
// Local row-major GEMM (standalone only)
// C[M,N] = opA(A)[M,K] * opB(B)[K,N]
// opA: trans_a? A^T : A
// opB: trans_b? B^T : B
// A,B,C : row-major contiguous
// ------------------------------
__global__ void k_gemm_rm(const float* __restrict__ A, const float* __restrict__ B,
                          float* __restrict__ C,
                          int M, int K, int N,
                          bool trans_a, bool trans_b)
{
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M || n >= N) return;

  float acc = 0.f;
  // Row-major indexing helpers
  // A_idx(m,k) if !trans_a, else A_idx(k,m) on original A
  for (int k=0;k<K;++k){
    int a_row = trans_a ? k : m;
    int a_col = trans_a ? m : k;
    int b_row = trans_b ? n : k;
    int b_col = trans_b ? k : n;
    float av = A[a_row * K + a_col];
    float bv = B[b_row * N + b_col];
    acc += av * bv;
  }
  C[m * N + n] = acc;
}

static inline Status gemm_local(const Tensor& A, const Tensor& B, Tensor& C,
                                bool trans_a, bool trans_b, StreamHandle s){
  // Preconditions: row-major contiguous 2D
  if (!A.is_contiguous_rowmajor_2d() || !B.is_contiguous_rowmajor_2d() || !C.is_contiguous_rowmajor_2d())
    return Status::LayoutMismatch;

  const int M = trans_a ? int(A.desc.shape[1]) : int(A.desc.shape[0]);
  const int K = trans_a ? int(A.desc.shape[0]) : int(A.desc.shape[1]);
  const int Kb= trans_b ? int(B.desc.shape[1]) : int(B.desc.shape[0]);
  const int N = trans_b ? int(B.desc.shape[0]) : int(B.desc.shape[1]);
  if (K != Kb) return Status::ShapeMismatch;

  dim3 bs(32, 8);
  dim3 gs(div_up(N, bs.x), div_up(M, bs.y));
  k_gemm_rm<<<gs,bs,0,(cudaStream_t)s>>>(A.data_ptr<const float>(), B.data_ptr<const float>(),
                                         C.data_ptr<float>(), M, K, N, trans_a, trans_b);
  return cudaGetLastError()==cudaSuccess ? Status::Ok : Status::RuntimeError;
}

// Standalone path: use local GEMM
static inline Status gemm_nn (const Tensor& A, const Tensor& B, Tensor& Y, StreamHandle s){ return gemm_local(A,B,Y,false,false,s); }
static inline Status gemm_tn (const Tensor& A_T, const Tensor& B, Tensor& Y, StreamHandle s){ return gemm_local(A_T,B,Y,true ,false,s); }
static inline Status gemm_nt (const Tensor& A, const Tensor& B_T, Tensor& Y, StreamHandle s){ return gemm_local(A,B_T,Y,false,true ,s); }

#else
// Integrated path: call core GEMM dispatcher
static inline Status gemm_nn(const Tensor& A, const Tensor& B, Tensor& Y, StreamHandle s){
  GemmAttrs ga; ga.trans_a=false; ga.trans_b=false; ga.act=ActKind::None; ga.with_bias=false; ga.leaky_slope=0.0f;
  int rc = ops::gemm_run(A,B,nullptr,Y,ga,s);
  return rc==0 ? Status::Ok : Status::RuntimeError;
}
static inline Status gemm_tn(const Tensor& A_T, const Tensor& B, Tensor& Y, StreamHandle s){
  GemmAttrs ga; ga.trans_a=true; ga.trans_b=false; ga.act=ActKind::None; ga.with_bias=false; ga.leaky_slope=0.0f;
  int rc = ops::gemm_run(A_T,B,nullptr,Y,ga,s);
  return rc==0 ? Status::Ok : Status::RuntimeError;
}
static inline Status gemm_nt(const Tensor& A, const Tensor& B_T, Tensor& Y, StreamHandle s){
  GemmAttrs ga; ga.trans_a=false; ga.trans_b=true; ga.act=ActKind::None; ga.with_bias=false; ga.leaky_slope=0.0f;
  int rc = ops::gemm_run(A,B_T,nullptr,Y,ga,s);
  return rc==0 ? Status::Ok : Status::RuntimeError;
}
#endif // BUILD_STANDALONE_OPS

// ===== Forward =====
Status RNNCudaLaunch(const Tensor& X, const Tensor& h0,
                     const Tensor& Wx, const Tensor& Wh,
                     const Tensor* b, Tensor& Hout, Tensor* Zbuf,
                     const RNNAttrs& attrs, StreamHandle s)
{
  const int T = attrs.T, B = attrs.B, I = attrs.I, H = attrs.H;
  if (T<=0 || B<=0 || I<=0 || H<=0) return Status::Invalid;

  // workspace (B,H) x2
  void* tmp1_ptr=nullptr; void* tmp2_ptr=nullptr;
  size_t bh_bytes = size_t(B) * size_t(H) * sizeof(float);
  if (cudaMalloc(&tmp1_ptr, bh_bytes) != cudaSuccess) return Status::RuntimeError;
  if (cudaMalloc(&tmp2_ptr, bh_bytes) != cudaSuccess){ cudaFree(tmp1_ptr); return Status::RuntimeError; }
  Tensor TMP1 = make_view2d(tmp1_ptr, B, H); // Z or temp
  Tensor TMP2 = make_view2d(tmp2_ptr, B, H); // temp

  Tensor hprev = h0; // [B,H], first step uses h0

  for (int t=0; t<T; ++t){
    const int64_t row0 = int64_t(t) * B;
    Tensor X_t  = row_slice_2d(X,    row0, B); // [B,I]
    Tensor H_t  = row_slice_2d(Hout, row0, B); // [B,H]
    Tensor Z_t  = (attrs.save_z && Zbuf) ? row_slice_2d(*Zbuf, row0, B) : TMP1;

    // Z_t = X_t @ Wx + hprev @ Wh + b
    AI_RETURN_IF_ERROR(gemm_nn(X_t,  Wx, Z_t, s));
    AI_RETURN_IF_ERROR(gemm_nn(hprev,Wh, TMP2, s));
    AI_RETURN_IF_ERROR(add_inplace(Z_t, TMP2, s));
    if (b) AI_RETURN_IF_ERROR(add_bias_rowwise(Z_t, *b, B, H, s));

    // H_t = tanh(Z_t)
    AI_RETURN_IF_ERROR(tanh_out(Z_t, H_t, s));

    hprev = H_t; // next step
  }

  cudaFree(tmp2_ptr);
  cudaFree(tmp1_ptr);
  return Status::Ok;
}

// ===== Backward =====
Status RNNCudaBackwardLaunch(const Tensor& X, const Tensor& Hout, const Tensor* Zbuf,
                             const Tensor& h0, const Tensor& Wx, const Tensor& Wh,
                             const Tensor& dHout,
                             Tensor* dX, Tensor* dh0, Tensor* dWx, Tensor* dWh, Tensor* dB,
                             const RNNAttrs& attrs, StreamHandle s)
{
  const int T = attrs.T, B = attrs.B, I = attrs.I, H = attrs.H;
  if (T<=0 || B<=0 || I<=0 || H<=0) return Status::Invalid;
  if (!dX || !dh0 || !dWx || !dWh || !dB) return Status::MissingInput;

  AI_RETURN_IF_ERROR(fill_zero(*dWx, s));
  AI_RETURN_IF_ERROR(fill_zero(*dWh, s));
  AI_RETURN_IF_ERROR(fill_zero(*dB,  s));

  void* tmp1_ptr=nullptr; void* tmp2_ptr=nullptr; void* dhn_ptr=nullptr;
  size_t bh_bytes = size_t(B) * size_t(H) * sizeof(float);
  if (cudaMalloc(&tmp1_ptr, bh_bytes) != cudaSuccess) return Status::RuntimeError;
  if (cudaMalloc(&tmp2_ptr, bh_bytes) != cudaSuccess){ cudaFree(tmp1_ptr); return Status::RuntimeError; }
  if (cudaMalloc(&dhn_ptr, bh_bytes) != cudaSuccess){ cudaFree(tmp1_ptr); cudaFree(tmp2_ptr); return Status::RuntimeError; }

  Tensor dHsum = make_view2d(tmp1_ptr, B, H);
  Tensor dZ    = make_view2d(tmp2_ptr, B, H);
  Tensor dh_next = make_view2d(dhn_ptr, B, H);
  AI_RETURN_IF_ERROR(fill_zero(dh_next, s));

  for (int t=T-1; t>=0; --t){
    const int64_t row0 = int64_t(t) * B;

    Tensor X_t   = row_slice_2d(X,     row0, B);
    Tensor dX_t  = row_slice_2d(*dX,   row0, B);
    Tensor H_t   = row_slice_2d(Hout,  row0, B);
    Tensor dH_t  = row_slice_2d(dHout, row0, B);
    (void)Zbuf;

    Tensor hprev = (t==0) ? h0 : row_slice_2d(Hout, (int64_t)(t-1)*B, B);

    AI_RETURN_IF_ERROR(add_out(dH_t, dh_next, dHsum, s));          // dHsum = dH_t + dh_next
    AI_RETURN_IF_ERROR(tanh_bwd_from_out(H_t, dHsum, dZ, s));      // dZ = dHsum * (1 - H^2)

    AI_RETURN_IF_ERROR(rowwise_sum_accum(dZ, *dB, B, H, s));       // db += sum_rows(dZ)
    AI_RETURN_IF_ERROR(gemm_tn(X_t,  dZ, *dWx, s));                // dWx += X^T @ dZ
    AI_RETURN_IF_ERROR(gemm_tn(hprev,dZ, *dWh, s));                // dWh += hprev^T @ dZ

    AI_RETURN_IF_ERROR(gemm_nt(dZ, Wx, dX_t, s));                  // dX_t  = dZ @ Wx^T
    AI_RETURN_IF_ERROR(gemm_nt(dZ, Wh, dh_next, s));               // dh_next = dZ @ Wh^T
  }

  AI_CUDA_TRY(cudaMemcpyAsync(dh0->data_ptr(), dh_next.data_ptr(), bh_bytes,
                              cudaMemcpyDeviceToDevice, (cudaStream_t)s));

  cudaFree(dhn_ptr);
  cudaFree(tmp2_ptr);
  cudaFree(tmp1_ptr);
  return Status::Ok;
}

} // namespace ai
