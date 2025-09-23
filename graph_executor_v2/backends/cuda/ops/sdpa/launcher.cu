#include <cuda_runtime.h>
#include <cmath>
#include "backends/cuda/ops/sdpa/api.hpp"
#include "backends/cuda/ops/dropout/api.hpp"
#include "ai/op_schema.hpp"
#include "ai/dispatch.hpp"

namespace ai { namespace ops {
  int gemm_run(const Tensor& A, const Tensor& B, const Tensor* Bias,
               Tensor& Y, const GemmAttrs& attrs, StreamHandle stream);
  int softmax_run(const Tensor& X, const Tensor* mask, Tensor& Y,
                  float scale, bool log, StreamHandle stream);
  int dropout_run(const Tensor& X, Tensor& Y, Tensor* mask,
                  const ai::DropoutAttrs& attrs, StreamHandle stream);
}}

namespace ai {

// --- helpers ---
static inline bool is_bhmd_f32_cuda_4d(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor && t.desc.shape.size()==4;
}
static inline cudaStream_t to_cuda(StreamHandle h){ return reinterpret_cast<cudaStream_t>(h); }

// 간단한 RowMajor 2D transpose 커널: in[R,C] -> out[C,R]
__global__ void transpose_rm_f32(const float* __restrict__ in, float* __restrict__ out,
                                 int R, int C)
{
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < R && c < C) {
    out[c * R + r] = in[r * C + c];
  }
}

Status SDPACudaLaunch(const Tensor& Q, const Tensor& K, const Tensor& V,
                      const Tensor* mask, Tensor& Y,
                      const SDPAAttrs& attrs, StreamHandle stream)
{
  if (!is_bhmd_f32_cuda_4d(Q) || !is_bhmd_f32_cuda_4d(K) ||
      !is_bhmd_f32_cuda_4d(V) || !is_bhmd_f32_cuda_4d(Y))
    return Status::Invalid;

  // shape: [B,H,M,D], [B,H,N,D], Y:[B,H,M,D]
  const int B  = (int)Q.desc.shape[0];
  const int H  = (int)Q.desc.shape[1];
  const int M  = (int)Q.desc.shape[2];
  const int D  = (int)Q.desc.shape[3];
  const int NB = (int)K.desc.shape[2]; // N

  if (K.desc.shape[0]!=B || K.desc.shape[1]!=H || K.desc.shape[3]!=D) return Status::ShapeMismatch;
  if (V.desc.shape[0]!=B || V.desc.shape[1]!=H || V.desc.shape[2]!=NB || V.desc.shape[3]!=D) return Status::ShapeMismatch;
  if (Y.desc.shape[0]!=B || Y.desc.shape[1]!=H || Y.desc.shape[2]!=M  || Y.desc.shape[3]!=D) return Status::ShapeMismatch;

  // 임시 텐서: S:[B,H,M,N], P:[B,H,M,N], Kt:[D,N] (슬라이스마다 재사용)
  size_t S_elems = (size_t)B*H*(size_t)M*(size_t)NB;
  float *dS=nullptr, *dP=nullptr, *dKt=nullptr;

  if (cudaMalloc(&dS, sizeof(float)*S_elems)!=cudaSuccess) return Status::RuntimeError;
  if (cudaMalloc(&dP, sizeof(float)*S_elems)!=cudaSuccess){ cudaFree(dS); return Status::RuntimeError; }
  // Kt는 한 슬라이스(D,N) 크기만 필요 (루프 내 재사용)
  if (cudaMalloc(&dKt, sizeof(float)*(size_t)D*(size_t)NB)!=cudaSuccess){ cudaFree(dS); cudaFree(dP); return Status::RuntimeError; }

  // Gemm 설정: transpose 플래그는 사용 안 함(항상 false)
  GemmAttrs g{};
  g.act        = ActKind::None;
  g.with_bias  = false;

  const float scale = (attrs.scale!=0.f) ? attrs.scale : (1.f/std::sqrt((float)D));
  cudaStream_t cstream = to_cuda(stream);

  // 슬라이스 도우미
  auto slice2d = [](const Tensor& T, int b, int h, int R, int C)->Tensor{
    size_t offset = ((size_t)b*T.desc.shape[1] + h) * (size_t)R*C;
    TensorDesc d{}; d.dtype=DType::F32; d.layout=Layout::RowMajor; d.shape={R,C}; d.stride={C,1};
    return Tensor{ (void*)((float*)T.data + offset), d, Device::CUDA, T.device_index };
  };

  auto S4 = Tensor{ dS, {DType::F32, Layout::RowMajor, {B,H,M,NB}, {H*M*NB, M*NB, NB, 1}}, Device::CUDA, Q.device_index };
  auto P4 = Tensor{ dP, {DType::F32, Layout::RowMajor, {B,H,M,NB}, {H*M*NB, M*NB, NB, 1}}, Device::CUDA, Q.device_index };

  // transpose launch config
  dim3 blk(32, 8);
  dim3 grd( (NB + blk.x - 1)/blk.x, (D + blk.y - 1)/blk.y );

  for (int b=0;b<B;++b){
    for (int h=0; h<H; ++h){
      Tensor Q2 = slice2d(Q, b,h,M,D);         // (M,D)
      Tensor K2 = slice2d(K, b,h,NB,D);        // (N,D)
      Tensor V2 = slice2d(V, b,h,NB,D);        // (N,D)

      // --- K(N,D) -> Kt(D,N) 전치 (연속 RowMajor로 보장) ---
      const float* Kptr = static_cast<const float*>(K2.data);
      transpose_rm_f32<<<grd, blk, 0, cstream>>>(Kptr, dKt, /*R*/NB, /*C*/D);
      if (cudaPeekAtLastError()!=cudaSuccess){ cudaFree(dKt); cudaFree(dS); cudaFree(dP); return Status::RuntimeError; }

      // Kt Tensor 래핑: [D, N], stride={N,1}
      TensorDesc kt_d{}; kt_d.dtype=DType::F32; kt_d.layout=Layout::RowMajor;
      kt_d.shape={D,NB}; kt_d.stride={NB,1};
      Tensor Kt{ dKt, kt_d, Device::CUDA, Q.device_index };

      // S = Q(M,D) @ Kt(D,N)
      Tensor S2 = slice2d(S4, b,h,M,NB);  // (M,N)
      {
        GemmAttrs gg = g;
        gg.trans_b = false;
        if (ai::ops::gemm_run(Q2, Kt, nullptr, S2, gg, stream)!=0){
          cudaFree(dKt); cudaFree(dS); cudaFree(dP); return Status::RuntimeError;
        }
      }

      // softmax(scale) → P2
      Tensor P2 = slice2d(P4, b,h,M,NB);
      const Tensor* m2 = nullptr; // (mask 브로드캐스트 향후)
      if (ai::ops::softmax_run(S2, m2, P2, /*scale*/scale, /*log*/false, stream)!=0){
        cudaFree(dKt); cudaFree(dS); cudaFree(dP); return Status::RuntimeError;
      }

      // (선택) dropout
      if (attrs.dropout_p > 0.f){
        ai::DropoutAttrs da{};
        da.p              = attrs.dropout_p;
        da.scale_in_train = attrs.scale_in_train;
        da.seed           = attrs.seed;
        if (ai::ops::dropout_run(P2, P2, /*mask*/nullptr, da, stream)!=0){
          cudaFree(dKt); cudaFree(dS); cudaFree(dP); return Status::RuntimeError;
        }
      }

      // Y = P(M,N) @ V(N,D)
      Tensor Y2 = slice2d(Y, b,h,M,D);
      {
        GemmAttrs gy = g;
        gy.trans_b = false;
        if (ai::ops::gemm_run(P2, V2, nullptr, Y2, gy, stream)!=0){
          cudaFree(dKt); cudaFree(dS); cudaFree(dP); return Status::RuntimeError;
        }
      }
    }
  }

  cudaFree(dKt);
  cudaFree(dS);
  cudaFree(dP);
  if (cudaPeekAtLastError()!=cudaSuccess) return Status::RuntimeError;
  return Status::Ok;
}

// Backward: 현재 미구현(링크 스텁)
Status SDPACudaBackwardLaunch(const Tensor& Q, const Tensor& K, const Tensor& V,
                              const Tensor* mask, const Tensor& dY,
                              Tensor* dQ, Tensor* dK, Tensor* dV,
                              const SDPAAttrs& attrs, StreamHandle stream)
{
  (void)Q; (void)K; (void)V; (void)mask; (void)dY;
  (void)dQ; (void)dK; (void)dV; (void)attrs; (void)stream;
  return Status::Unimplemented;
}

} // namespace ai
