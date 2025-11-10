// backends/cuda/ops/conv2d/launcher.cu
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cassert>

#include "backends/cuda/ops/conv2d/api.hpp"
#include "backends/cuda/ops/gemm/api.hpp"   // GemmCudaLaunch / GemmCudaBackward (epilogue 사용)
#include "backends/cuda/ops/_common/shim/ai_shim.hpp"


namespace ai {

// ===== utils & externs =====
__global__ void kadd_kernel(float* A, const float* B, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) A[i] += B[i];
}
// transpose(src[K,Cout]) and add into dst[Cout,K]  (dst += src^T)
__global__ void kadd_transpose_kernel(const float* __restrict__ srcKC,
                                      float* __restrict__ dstCK,
                                      int K, int Cout)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int c = blockIdx.y * blockDim.y + threadIdx.y;
  if (k < K && c < Cout) {
    float v = srcKC[(size_t)k * Cout + c];
    atomicAdd(&dstCK[(size_t)c * K + k], v);
  }
}

static inline bool is4_f32_cuda(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor && t.desc.shape.size()==4;
}
static inline bool is1_f32_cuda(const Tensor& t){
  return t.device==Device::CUDA && t.desc.dtype==DType::F32 &&
         t.desc.layout==Layout::RowMajor && t.desc.shape.size()==1;
}
static inline cudaStream_t to_cuda(StreamHandle h){ return reinterpret_cast<cudaStream_t>(h); }

// im2col/col2im/transpose 런처 (선언부)
void im2col_kernel_launcher(const float*, float*,
                            int,int,int, int,int, int,int, int,int, int,int, int,int, cudaStream_t);
void col2im_kernel_launcher(const float*, float*,
                            int,int,int, int,int, int,int, int,int, int,int, int,int, cudaStream_t);
// row-major transpose: in[M,N] -> out[N,M]
void transpose_kernel_launcher(const float* A, float* AT, int M, int N, cudaStream_t);

// ===== dB reduce (co-by-row) [남겨두되, 현재 구현에서는 사용하지 않음] =====
__global__ void reduce_db_rows_kernel(const float* __restrict__ gy, // [Cout, HWo]
                                      float* __restrict__ db,       // [Cout]
                                      int Cout, int HWo)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = Cout * HWo;
  if (idx >= total) return;
  int hw = idx % HWo;
  int co = idx / HWo;
  atomicAdd(&db[co], gy[(size_t)co * HWo + hw]);
}
static inline void reduce_db_rows_kernel_launcher(const float* gy, float* db, int Cout, int HWo, cudaStream_t s){
  const int total = Cout * HWo;
  constexpr int BS = 256;
  dim3 block(BS), grid((total + BS - 1)/BS);
  reduce_db_rows_kernel<<<grid, block, 0, s>>>(gy, db, Cout, HWo);
}

// ===== pack/unpack W =====
__global__ void pack_w_oihw_to_KC(const float* __restrict__ W, float* __restrict__ out_KC,
                                  int Cout, int Cin, int Kh, int Kw);
__global__ void pack_w_oihw_to_CK(const float* __restrict__ W, float* __restrict__ out_CK,
                                  int Cout, int Cin, int Kh, int Kw);
__global__ void unpack_ck_to_oihw_add(const float* __restrict__ dWpack, float* __restrict__ dW,
                                      int Cout, int Cin, int Kh, int Kw);

// ===== activation backward (elementwise) on [Cout, HWo] [이전 경로—현재 미사용] =====
__device__ __forceinline__ float dact(ai::ActKind act, float z, float gy, float slope) {
  switch (act) {
    case ai::ActKind::None:    return gy;
    case ai::ActKind::ReLU:    return (z > 0.f) ? gy : 0.f;
    case ai::ActKind::LeakyReLU:return (z > 0.f) ? gy : slope * gy;
    case ai::ActKind::Sigmoid: {
      float s = 1.f / (1.f + __expf(-z));
      return gy * s * (1.f - s);
    }
    case ai::ActKind::Tanh: {
      float t = tanhf(z);
      return gy * (1.f - t*t);
    }
    case ai::ActKind::GELU: {
      const float c = sqrtf(2.f / 3.1415926535f);
      float z3 = z*z*z;
      float th = tanhf(c*(z + 0.044715f*z3));
      float dtanh = (1 - th*th) * c * (1 + 0.134145f*z*z);
      return gy * (0.5f*(1 + th) + 0.5f*z*dtanh);
    }
    default: return gy;
  }
}
__global__ void apply_dact_rows(const float* __restrict__ gy_post,
                                const float* __restrict__ Z_rows,
                                float* __restrict__ gy_rows,
                                int Cout, int HWo, ai::ActKind act, float slope)
{
  int hw = blockIdx.x * blockDim.x + threadIdx.x;
  int co = blockIdx.y * blockDim.y + threadIdx.y;
  if (co < Cout && hw < HWo) {
    size_t idx = (size_t)co * HWo + hw;
    gy_rows[idx] = dact(act, Z_rows[idx], gy_post[idx], slope);
  }
}

// ======================= Forward (no cudaMalloc) =======================
Status Conv2DCudaLaunch(const Tensor& X, const Tensor& W, const Tensor* B, Tensor& Y,
                        const Conv2DAttrs& a, StreamHandle stream, Tensor* Z_saved,
                        const Conv2DWorkspaceFwd* ws_fwd)
{
  if (!is4_f32_cuda(X) || !is4_f32_cuda(Y)) return Status::Invalid;
  if (!is4_f32_cuda(W)) return Status::Invalid;
  if (a.groups != 1)    return Status::Unimplemented;
  if (!ws_fwd || !ws_fwd->dCol || !ws_fwd->W_KC || !ws_fwd->Y_tmp)
    return Status::MissingInput; // workspace 부족

  const int N   = (int)X.desc.shape[0];
  const int Cin = (int)X.desc.shape[1];
  const int H   = (int)X.desc.shape[2];
  const int Wd  = (int)X.desc.shape[3];

  const int Cout= (int)W.desc.shape[0];
  const int WCin= (int)W.desc.shape[1];
  const int Kh  = (int)W.desc.shape[2];
  const int Kw  = (int)W.desc.shape[3];
  if (WCin != Cin) return Status::ShapeMismatch;

  const int Ho = (H  + 2*a.pad_h - a.dil_h*(Kh-1) - 1)/a.stride_h + 1;
  const int Wo = (Wd + 2*a.pad_w - a.dil_w*(Kw-1) - 1)/a.stride_w + 1;

  if (Y.desc.shape[0]!=N || Y.desc.shape[1]!=Cout || Y.desc.shape[2]!=Ho || Y.desc.shape[3]!=Wo)
    return Status::ShapeMismatch;

  if (N<=0 || Cin<=0 || H<=0 || Wd<=0 || Cout<=0 || Kh<=0 || Kw<=0 || Ho<=0 || Wo<=0)
    return Status::Invalid;

  // Z_saved 체크
  const bool want_z = a.save_z && (Z_saved != nullptr);
  if (want_z) {
    if (!is4_f32_cuda(*Z_saved)) return Status::Invalid;
    if (Z_saved->desc.shape[0]!=N || Z_saved->desc.shape[1]!=Cout ||
        Z_saved->desc.shape[2]!=Ho || Z_saved->desc.shape[3]!=Wo) return Status::ShapeMismatch;
  }

  const float* dW = static_cast<const float*>(W.data);
  const float* dB = (B && B->data && a.with_bias) ? static_cast<const float*>(B->data) : nullptr;

  const int K   = Cin*Kh*Kw;
  const int HWo = Ho*Wo;
  auto s = to_cuda(stream);

  float* dCol   = ws_fwd->dCol;            // [HWo, K]
  float* W_KC   = ws_fwd->W_KC;            // [K, Cout]
  float* Y_tmp  = ws_fwd->Y_tmp;           // [HWo, Cout]
  float* Z_rows = ws_fwd->Z_rows;          // [HWo, Cout] or nullptr

  if (want_z && !Z_rows) return Status::MissingInput; // pre-act rows 필요

  // pack W → [K, Cout]
  {
    dim3 block(256), grid((K + block.x - 1)/block.x, Cout);
    pack_w_oihw_to_KC<<<grid, block, 0, s>>>(dW, W_KC, Cout, Cin, Kh, Kw);
  }

  // GEMM attrs: epilogue 사용
  ai::GemmAttrs g{};
  g.act         = a.act;          // Conv2D와 Gemm 모두 ActKind
  g.leaky_slope = a.leaky_slope;
  g.with_bias   = (dB != nullptr);
  g.save_z      = want_z;

  for (int n=0; n<N; ++n) {
    const float* x_n = static_cast<const float*>(X.data) + (size_t)n*Cin*H*Wd;
    float*       y_n = static_cast<float*>(Y.data)       + (size_t)n*Cout*Ho*Wo;
    float*       z_n = want_z ? static_cast<float*>(Z_saved->data) + (size_t)n*Cout*Ho*Wo : nullptr;

    // im2col: [HWo,K]
    im2col_kernel_launcher(
      x_n, dCol,
      Cin, H, Wd, Kh, Kw,
      a.stride_h, a.stride_w, a.pad_h, a.pad_w, a.dil_h, a.dil_w,
      Ho, Wo, s
    );

    // GEMM rows
    Tensor tA{dCol,  {DType::F32, Layout::RowMajor, {HWo, K},    {K, 1}},     Device::CUDA, 0};
    Tensor tB{W_KC,  {DType::F32, Layout::RowMajor, {K,   Cout}, {Cout, 1}},  Device::CUDA, 0};
    Tensor tY{Y_tmp, {DType::F32, Layout::RowMajor, {HWo, Cout}, {Cout, 1}},  Device::CUDA, 0};

    Tensor tZcap{};  // [HWo,Cout] pre-act rows
    const ai::Tensor* BiasPtr = nullptr;

    if (want_z) {
      tZcap = Tensor{Z_rows, {DType::F32, Layout::RowMajor, {HWo, Cout}, {Cout,1}}, Device::CUDA, 0};
    }
    ai::Tensor BiasT;
    if (dB) {
      BiasT.data         = const_cast<float*>(dB);
      BiasT.device       = ai::Device::CUDA;
      BiasT.device_index = 0;
      BiasT.desc.dtype   = ai::DType::F32;
      BiasT.desc.layout  = ai::Layout::RowMajor;
      BiasT.desc.shape   = { (int64_t)Cout };
      BiasT.desc.stride  = { 1 };
      BiasPtr = &BiasT;
    }

    ai::Status st = ai::GemmCudaLaunch(
        tA, tB, BiasPtr, tY, g, stream,
        (want_z ? &tZcap : nullptr)
    );
    if (st != ai::Status::Ok) return st;

    // rows -> NCHW
    transpose_kernel_launcher(Y_tmp, y_n, /*M=*/HWo, /*N=*/Cout, s);
    if (want_z) {
      transpose_kernel_launcher(Z_rows, z_n, /*M=*/HWo, /*N=*/Cout, s);
    }
  }

  return Status::Ok;
}

// ======================= Backward (no cudaMalloc) =======================
// 이번 수정: (1) 활성화 미분/바이어스 축약/중간 gZ 생성은 GemmCudaBackward의 에필로그로 처리
//           (2) gA_rows(HWo,K) → col2im → dX
//           (3) gB_KC(K,Cout)   를 전치 추가(accumulate)하여 dWpack[Cout,K]에 누적 후, 최종 unpack
Status Conv2DCudaBackwardLaunch(const Tensor& X, const Tensor& W, const Tensor& dY_post,
                                const Tensor& Z, Tensor* dW, Tensor* dB, Tensor* dX,
                                const Conv2DAttrs& a, StreamHandle stream,
                                const Conv2DWorkspaceBwd* ws_bwd)
{
  if (!is4_f32_cuda(X) || !is4_f32_cuda(W) || !is4_f32_cuda(dY_post) || !is4_f32_cuda(Z)) return Status::Invalid;
  if (a.groups != 1) return Status::Unimplemented;
  if (!ws_bwd || !ws_bwd->dCol || !ws_bwd->dTmp || !ws_bwd->dWpack || !ws_bwd->gy_rows || !ws_bwd->Z_rows || !ws_bwd->dY_HT)
    return Status::MissingInput; // dCol, dTmp(다목적), dWpack(Cout*K), gy_rows(HWo*Cout), Z_rows(HWo*Cout), dY_HT(gZ scratch)

  const int N   = (int)X.desc.shape[0];
  const int Cin = (int)X.desc.shape[1];
  const int H   = (int)X.desc.shape[2];
  const int Wd  = (int)X.desc.shape[3];

  const int Cout= (int)W.desc.shape[0];
  const int WCin= (int)W.desc.shape[1];
  const int Kh  = (int)W.desc.shape[2];
  const int Kw  = (int)W.desc.shape[3];
  if (WCin != Cin) return Status::ShapeMismatch;

  const int Ho = (H  + 2*a.pad_h - a.dil_h*(Kh-1) - 1)/a.stride_h + 1;
  const int Wo = (Wd + 2*a.pad_w - a.dil_w*(Kw-1) - 1)/a.stride_w + 1;

  if (dY_post.desc.shape[0]!=N || dY_post.desc.shape[1]!=Cout || dY_post.desc.shape[2]!=Ho || dY_post.desc.shape[3]!=Wo)
    return Status::ShapeMismatch;
  if (Z.desc.shape[0]!=N || Z.desc.shape[1]!=Cout || Z.desc.shape[2]!=Ho || Z.desc.shape[3]!=Wo)
    return Status::ShapeMismatch;

  if (dW) {
    if (!is4_f32_cuda(*dW) ||
        dW->desc.shape[0]!=Cout || dW->desc.shape[1]!=Cin ||
        dW->desc.shape[2]!=Kh   || dW->desc.shape[3]!=Kw) return Status::ShapeMismatch;
  }
  if (dB) { if (!(is1_f32_cuda(*dB) && (int)dB->desc.shape[0]==Cout)) return Status::ShapeMismatch; }
  if (dX) {
    if (!is4_f32_cuda(*dX) ||
        dX->desc.shape[0]!=N || dX->desc.shape[1]!=Cin ||
        dX->desc.shape[2]!=H || dX->desc.shape[3]!=Wd) return Status::ShapeMismatch;
  }

  if (N<=0 || Cin<=0 || H<=0 || Wd<=0 || Cout<=0 || Kh<=0 || Kw<=0 || Ho<=0 || Wo<=0)
    return Status::Invalid;

  const int K   = Cin*Kh*Kw;
  const int HWo = Ho*Wo;
  auto s = to_cuda(stream);

  // 워크스페이스 별칭
  float* dCol    = ws_bwd->dCol;     // [HWo, K]  — im2col 입력 및 (BWD 후) gA_rows로 재사용
  float* dTmp    = ws_bwd->dTmp;     // 다목적 버퍼 — 여기서는 gB_KC [K,Cout] 보관에 사용
  float* dWpack  = ws_bwd->dWpack;   // [Cout, K] — 누적 저장
  float* gy_rows = ws_bwd->gy_rows;  // [HWo, Cout]
  float* Z_rows  = ws_bwd->Z_rows;   // [HWo, Cout]
  float* gZ_scratch = ws_bwd->dY_HT; // [HWo, Cout] — GemmCudaBackward의 gZ scratch 용도로 재사용

  // zero grads (안전)
  if (dB) cudaMemsetAsync(dB->data, 0, sizeof(float)*Cout, s);
  if (dW) {
    cudaMemsetAsync(dW->data, 0, sizeof(float)*Cout*Cin*Kh*Kw, s);
    cudaMemsetAsync(dWpack,   0, sizeof(float)*Cout*K, s);
  }
  if (dX) cudaMemsetAsync(dX->data, 0, sizeof(float)*N*Cin*H*Wd, s);

  // BWD에서 사용할 W_KC 버퍼: dTmp가 (K*Cout) 이상이면 사용, 아니면 pack용 별도 한 번 더 할당 필요
  // 여기서는 pack을 매 스텝 전 덮어쓰는 형태로 dTmp를 W_KC로 사용 → BWD 호출 시 gB는 dTmp와는 다른 포인터여야 하므로
  // pack은 루프 밖에서 한 번 수행하고, 루프 안에서는 dTmp를 gB_KC로 사용.
  {
    const float* dWsrc = static_cast<const float*>(W.data);
    dim3 block(256), grid((K + block.x - 1)/block.x, Cout);
    pack_w_oihw_to_KC<<<grid, block, 0, s>>>(dWsrc, dTmp /*W_KC*/, Cout, Cin, Kh, Kw);
  }
  float* W_KC = dTmp; // [K, Cout], 루프 내에서 변경하지 않음

  // GEMM attrs (BWD에서 act/backprop 에필로그 사용)
  ai::GemmAttrs g{}; 
  g.act         = a.act;
  g.leaky_slope = a.leaky_slope;
  g.with_bias   = false; // gBias는 출력으로 직접 받음

  // GemmWorkspace: gZ scratch 전달(필수)
  ai::GemmWorkspace gws{};
  gws.lt_workspace       = nullptr;
  gws.lt_workspace_bytes = 0;
  gws.scratch            = gZ_scratch;
  gws.scratch_bytes      = (size_t)HWo * (size_t)Cout * sizeof(float);

  for (int n=0; n<N; ++n) {
    const float* x_n   = static_cast<const float*>(X.data)      + (size_t)n*Cin*H*Wd;
    const float* gy_nP = static_cast<const float*>(dY_post.data)+ (size_t)n*Cout*Ho*Wo; // post-act grad
    const float* z_n   = static_cast<const float*>(Z.data)      + (size_t)n*Cout*Ho*Wo;

    // NCHW → rows([HWo,Cout]) 뷰
    {
      size_t rows_bytes = sizeof(float) * (size_t)HWo * Cout;
      // gy_rows: [HWo,Cout] (RowMajor)
      // Z_rows : [HWo,Cout] (RowMajor)
      transpose_kernel_launcher(gy_nP, gy_rows, /*M=*/Cout, /*N=*/HWo, s); // from [Cout,HWo] to [HWo,Cout]
      transpose_kernel_launcher(z_n,   Z_rows,  /*M=*/Cout, /*N=*/HWo, s);
    }

    // im2col(X[n]) → dCol [HWo,K]
    im2col_kernel_launcher(
      x_n, dCol,
      Cin, H, Wd, Kh, Kw,
      a.stride_h, a.stride_w, a.pad_h, a.pad_w, a.dil_h, a.dil_w,
      Ho, Wo, s
    );

    // 준비된 텐서 뷰
    Tensor tA{ dCol,    {DType::F32, Layout::RowMajor, {HWo, K},    {K,    1}}, Device::CUDA, 0};  // A rows
    Tensor tB{ W_KC,    {DType::F32, Layout::RowMajor, {K,   Cout}, {Cout, 1}}, Device::CUDA, 0};  // W_KC
    Tensor tgY{ gy_rows,{DType::F32, Layout::RowMajor, {HWo, Cout}, {Cout, 1}}, Device::CUDA, 0};  // dY_post rows
    Tensor tZ{  Z_rows, {DType::F32, Layout::RowMajor, {HWo, Cout}, {Cout, 1}}, Device::CUDA, 0};  // Z rows

    // 출력: gA_rows(HWo,K) — dCol 버퍼를 그대로 덮어쓰기 활용
    Tensor t_gA{ dCol,  {DType::F32, Layout::RowMajor, {HWo, K},    {K,    1}}, Device::CUDA, 0};
    // 출력: gB_KC(K,Cout) — dTmp를 루프 내에서는 gB_KC 용도로 재사용 (W_KC는 루프 밖에서 채워짐, 동일 포인터 재사용 금지)
    // → 따라서 W_KC는 루프 시작 전에 dTmp에 만들어 두었고, 여기서는 gB_KC로 dTmp를 덮어쓰면 안 되므로
    //    gB_KC 용으로는 별도의 영역이 필요. dTmp를 W_KC로 계속 유지하려면, gB_KC는 다른 버퍼가 필요하다.
    //    여기서는 gy_rows 버퍼가 이후 col2im 전까지는 사용되지 않으므로 gB_KC 임시로 gy_rows를 재활용하지 않는다(shape 다름).
    //    안전하게 dTmp를 gB_KC로 사용하고, 전 단계의 W_KC는 루프 밖 한 번만 pack했으니 여기서 덮어써도 문제 없음.
    //    => 루프 안에서 W_KC가 필요하므로, W_KC는 별도 버퍼가 필요. ws_bwd에 W_KC 필드를 두지 않았다면,
    //       전 단계에서 dTmp에 pack한 뒤, 바로 별도 버퍼로 복사해 두자.
  }

  // ---- 재정렬: 루프 밖에서 사용할 버퍼 조정 ----
  // 위 루프 구현을 단순화하기 위해, 루프 시작 전에 W_KC를 dTmp에 pack하고,
  // 바로 ws_bwd->W_CK 자리에 transpose(W_KC)하여 보관해 두고(있다면),
  // 루프 안에서는 항상 그 보관 버퍼로부터 다시 전치하지 않고 사용하도록 바꿀 수 있다.
  // 다만 현재 ws_bwd 정의를 바꾸지 않고 진행하기 위해 아래와 같이 구현을 마저 완성한다.

  // === 실제 구현(루프 포함) 재작성 ===
  {
    // W_KC 별도 보관 버퍼가 없으므로, 여기서 한 번 더 준비: 전용 버퍼가 없다고 가정하면
    // dTmp를 두 용도로 공유할 수 없기 때문에 루프를 다시 작성한다.
  }

  // ---- 최종 정리: 위의 제약을 해소한 실제 동작 루프 (완성본) ----
  // 가정: ws_bwd에 다음 추가 버퍼를 사용할 수 있다:
  //  - W_KC_buf : [K,Cout]  (없으면 dTmp를 사용하고, 루프마다 pack 수행)
  // 본 구현은 dTmp를 gB_KC로, 별도의 W_KC_buf를 W_KC 보관용으로 사용한다.
  // 만약 W_KC_buf가 없다면, 루프 시작 시마다 pack_w_oihw_to_KC를 호출해 dTmp에 W_KC를 만들고,
  // 즉시 gB 호출에 사용할 다른 버퍼가 필요하므로 구조가 복잡해진다. 아래는 W_KC_buf가 있다고 가정한 경량 경로.

  // ---- 안전한 최종 버전: W_KC_buf 필드가 있다고 가정 (권장) ----
  // 필드 이름은 기존 구조를 존중해 ws_bwd->W_CK 를 재사용(실제 레이아웃은 KC로 채운다).
  if (!ws_bwd->W_CK) return Status::MissingInput; // 재사용 버퍼 필요
  float* W_KC_buf = ws_bwd->W_CK; // [Cout,K] 크기로 잡혀 있을 수 있으나, 메모리는 K*Cout 만큼이면 OK. 우리는 [K,Cout]로 사용.

  {
    const float* dWsrc = static_cast<const float*>(W.data);
    dim3 block(256), grid((K + block.x - 1)/block.x, Cout);
    pack_w_oihw_to_KC<<<grid, block, 0, s>>>(dWsrc, W_KC_buf /*[K,Cout]*/, Cout, Cin, Kh, Kw);
  }

  for (int n=0; n<N; ++n) {
    const float* x_n   = static_cast<const float*>(X.data)      + (size_t)n*Cin*H*Wd;
    const float* gy_nP = static_cast<const float*>(dY_post.data)+ (size_t)n*Cout*Ho*Wo; // [Cout,HWo]
    const float* z_n   = static_cast<const float*>(Z.data)      + (size_t)n*Cout*Ho*Wo; // [Cout,HWo]

    // rows 뷰로 변환 (NCHW → [HWo,Cout])
    transpose_kernel_launcher(gy_nP, gy_rows, /*M=*/Cout, /*N=*/HWo, s);
    transpose_kernel_launcher(z_n,   Z_rows,  /*M=*/Cout, /*N=*/HWo, s);

    // im2col(X[n]) → dCol [HWo,K]
    im2col_kernel_launcher(
      x_n, dCol,
      Cin, H, Wd, Kh, Kw,
      a.stride_h, a.stride_w, a.pad_h, a.pad_w, a.dil_h, a.dil_w,
      Ho, Wo, s
    );

    // 뷰 준비
    Tensor tA{ dCol,     {DType::F32, Layout::RowMajor, {HWo, K},    {K,    1}}, Device::CUDA, 0}; // A rows
    Tensor tB{ W_KC_buf, {DType::F32, Layout::RowMajor, {K,   Cout}, {Cout, 1}}, Device::CUDA, 0}; // W_KC
    Tensor tgY{ gy_rows, {DType::F32, Layout::RowMajor, {HWo, Cout}, {Cout, 1}}, Device::CUDA, 0};
    Tensor tZ{  Z_rows,  {DType::F32, Layout::RowMajor, {HWo, Cout}, {Cout, 1}}, Device::CUDA, 0};

    // 출력: gA_rows(HWo,K) — dCol 버퍼로 덮어쓰기
    Tensor t_gA{ dCol,   {DType::F32, Layout::RowMajor, {HWo, K},    {K,    1}}, Device::CUDA, 0};
    // 출력: gB_KC(K,Cout) — dTmp 버퍼 사용
    Tensor t_gB{ dTmp,   {DType::F32, Layout::RowMajor, {K,   Cout}, {Cout, 1}}, Device::CUDA, 0};
    // gBias 출력
    Tensor t_gBias{};
    if (dB && a.with_bias) {
      t_gBias.data         = dB->data;
      t_gBias.device       = ai::Device::CUDA;
      t_gBias.device_index = 0;
      t_gBias.desc = { ai::DType::F32, ai::Layout::RowMajor, { (int64_t)Cout }, { 1 } };
    }

    // GemmCudaBackward: epilogue( act' , gBias reduce ) + gA/gB 계산
    ai::Status st = ai::GemmCudaBackward(
      tA, tB, /*C*/nullptr,
      tgY, tZ,
      /*gA*/ &t_gA, /*gB*/ &t_gB, /*gC*/ nullptr, /*gBias*/ (dB && a.with_bias ? &t_gBias : nullptr),
      g, stream, &gws);
    if (st != ai::Status::Ok) return st;

    // gA_rows(HWo,K)=dCol → col2im → dX[n]
    if (dX) {
      float* dx_n = static_cast<float*>(dX->data) + (size_t)n*Cin*H*Wd;
      col2im_kernel_launcher(
        dCol, dx_n,
        Cin, H, Wd, Kh, Kw,
        a.stride_h, a.stride_w, a.pad_h, a.pad_w, a.dil_h, a.dil_w,
        Ho, Wo, s
      );
    }

    // gB_KC(K,Cout)=dTmp  →  dWpack[Cout,K] 에 전치-가산
    if (dW) {
      dim3 blk(32, 8);
      dim3 grd( (K + blk.x - 1)/blk.x, (Cout + blk.y - 1)/blk.y );
      kadd_transpose_kernel<<<grd, blk, 0, s>>>(dTmp, dWpack, K, Cout);
    }
  }

  // dWpack[Cout,K] -> dW[O,I,H,W]
  if (dW) {
    dim3 block(256), grid((K + block.x - 1)/block.x, Cout);
    unpack_ck_to_oihw_add<<<grid, block, 0, s>>>(dWpack, static_cast<float*>(dW->data), Cout, Cin, Kh, Kw);
  }

  return Status::Ok;
}

} // namespace ai
