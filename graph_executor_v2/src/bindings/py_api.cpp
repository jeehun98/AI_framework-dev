// bindings/py_api.cpp
#include <stdexcept>
#include <string>
#include <vector>
#include <cctype>
#include <cstdint>
#include <algorithm>
#include <optional>

#include <cuda_runtime.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "ai/tensor.hpp"
#include "ai/dispatch.hpp"
#include "ai/op_schema.hpp"
#include "regemm/api.h"  // EX forward(Z stash) 직접 호출용
#include "backends/cuda/ops/rmsnorm/api.hpp"  // ✅ RMSNormAttrs, 런처 시그니처 출처 통일
#include "backends/cuda/ops/layernorm/api.hpp"
#include "backends/cuda/ops/softmax/api.hpp"
#include "backends/cuda/ops/cross_entropy/api.hpp"
#include "backends/cuda/ops/dropout/api.hpp"
#include "backends/cuda/ops/sdpa/api.hpp"  
#include "backends/cuda/ops/conv2d/api.hpp"  
#include "backends/cuda/ops/pool2d/api.hpp"
#include "backends/cuda/ops/elementwise/api.hpp"
#include "backends/cuda/ops/reduction/api.hpp"



namespace py = pybind11;
using namespace ai;

// --- 외부 등록 함수: import 시 반드시 1회 호출 ---
extern "C" void ai_backend_cuda_register_all();

// --- CUDA helpers ---
static inline void checkCuda(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
  }
}

// numpy-style shape 계산
static std::vector<int64_t> reduced_shape(const std::vector<int64_t>& shp,
                                          const std::vector<int>& axes,
                                          bool keepdim)
{
  std::vector<int64_t> out;
  const int nd = (int)shp.size();
  std::vector<char> is_ax(nd, 0);
  for (int a: axes) {
    int aa=a;
    if (aa<0) aa+=nd;
    is_ax[aa]=1;
  }

  if (keepdim){
    out = shp;
    for (int i=0;i<nd;i++) if (is_ax[i]) out[i]=1;
  } else {
    for (int i=0;i<nd;i++) if (!is_ax[i]) out.push_back(shp[i]);
    if (out.empty()) out.push_back(1); // all-reduce → scalar [1]
  }
  return out;
}

static inline ActKind parse_act(const std::string& s){
  std::string k; k.reserve(s.size());
  for (char c: s) k.push_back(std::tolower(static_cast<unsigned char>(c)));
  if (k=="none" || k=="identity") return ActKind::None;
  if (k=="relu")                  return ActKind::ReLU;
  if (k=="leakyrelu" || k=="leaky_relu" || k=="lrelu") return ActKind::LeakyReLU;
  if (k=="gelu")                  return ActKind::GELU;
  if (k=="sigmoid")               return ActKind::Sigmoid;
  if (k=="tanh")                  return ActKind::Tanh;
  throw std::runtime_error("unknown activation: " + s);
}

static inline regemm::ActKind to_regemm_act(ActKind a) {
  switch (a) {
    case ActKind::None:      return regemm::ActKind::None;
    case ActKind::ReLU:      return regemm::ActKind::ReLU;
    case ActKind::LeakyReLU: return regemm::ActKind::LeakyReLU;
    case ActKind::GELU:      return regemm::ActKind::GELU;
    case ActKind::Sigmoid:   return regemm::ActKind::Sigmoid;
    case ActKind::Tanh:      return regemm::ActKind::Tanh;
  }
  return regemm::ActKind::None;
}
static inline TensorDesc make_desc_2d(int64_t rows, int64_t cols){
  TensorDesc d{};
  d.dtype  = DType::F32;
  d.layout = Layout::RowMajor;
  d.shape  = {rows, cols};
  d.stride = {cols, 1};
  return d;
}

static inline ai::TensorDesc make_desc_2d_any(int64_t r, int64_t c, ai::DType dt){
  ai::TensorDesc d{};
  d.dtype  = dt;
  d.layout = ai::Layout::RowMajor;
  d.shape  = {r, c};
  d.stride = {c, 1};
  return d;
}

static inline ai::TensorDesc make_desc_1d_any(int64_t n, ai::DType dt){
  ai::TensorDesc d{};
  d.dtype  = dt;
  d.layout = ai::Layout::RowMajor;
  d.shape  = {n};
  d.stride = {1};
  return d;
}

static inline ai::TensorDesc make_desc_1d_f32(int64_t n){
  ai::TensorDesc d{};
  d.dtype  = ai::DType::F32;
  d.layout = ai::Layout::RowMajor;
  d.shape  = {n};
  d.stride = {1};
  return d;
}

// ================== ⬇️⬇️⬇️ 추가: 풀2D 출력 크기 (정식 공식) ⬇️⬇️⬇️ ==================
static inline int pool2d_out_dim_host(int H_in, int k, int s, int p, int d, bool ceil_mode) {
  // floor/ceil((H_in + 2p - effK)/s) + 1
  const int eff = (k - 1) * d + 1;
  const int a   = H_in + 2 * p - eff;
  if (a < 0) return 0;
  if (ceil_mode) {
    return ( (a + s - 1) / s ) + 1;
  } else {
    return ( a / s ) + 1;
  }
}

static inline void pool2d_output_dims_host(
    int H, int W, int kH, int kW, int sH, int sW,
    int pH, int pW, int dH, int dW, bool ceil_mode,
    int& Ho, int& Wo) {
  Ho = pool2d_out_dim_host(H, kH, sH, pH, dH, ceil_mode);
  Wo = pool2d_out_dim_host(W, kW, sW, pW, dW, ceil_mode);
  if (Ho < 0) Ho = 0;
  if (Wo < 0) Wo = 0;
}
// ================== ⬆️⬆️⬆️ 추가 끝 ⬆️⬆️⬆️ ==================

// -------------------- 디스패치 엔트리(정의는 src/dispatch/registry.cpp) --------------------
namespace ai { namespace ops {
using ::ai::LayerNormAttrs;

int gemm_run(const Tensor& A, const Tensor& B, const Tensor* Bias,
             Tensor& Y, const GemmAttrs& attrs, StreamHandle stream);

int gemm_bwd_run(const Tensor& A, const Tensor& B, const Tensor* C,
                 const Tensor& gY, const Tensor& Z,
                 Tensor* gA, Tensor* gB, Tensor* gC, Tensor* gBias,
                 const GemmAttrs& attrs, StreamHandle stream);

// ⬇️ RMSNorm은 attrs 타입이 ai::RMSNormAttrs 임에 유의
int rmsnorm_run(const Tensor&, const Tensor*, const Tensor*, Tensor&, const ai::RMSNormAttrs&, StreamHandle);
int rmsnorm_backward_run(const Tensor&, const Tensor*, const Tensor&, Tensor&, Tensor*, Tensor*, const ai::RMSNormAttrs&, StreamHandle);

int layernorm_run(const Tensor&, const Tensor*, const Tensor*, Tensor&, const ai::LayerNormAttrs&, StreamHandle);
int layernorm_backward_run(const Tensor&, const Tensor*, const Tensor&, Tensor&, Tensor*, Tensor*, const ai::LayerNormAttrs&, StreamHandle);

int softmax_run(const Tensor&, const Tensor*, Tensor&, const ai::SoftmaxAttrs&, StreamHandle);
int softmax_backward_run(const Tensor& Y, const Tensor& dY, Tensor& dX, const ai::SoftmaxAttrs&, StreamHandle);

int cross_entropy_run(const Tensor&, const Tensor&, Tensor&, const ai::CrossEntropyAttrs&, StreamHandle);
int cross_entropy_backward_run(const Tensor&, const Tensor&, Tensor&, const ai::CrossEntropyAttrs&, StreamHandle);

int dropout_run(const Tensor& X, Tensor& Y, Tensor* mask, const ai::DropoutAttrs& attrs, StreamHandle);
int dropout_backward_run(const Tensor& dY, const Tensor& mask, Tensor& dX, const ai::DropoutAttrs& attrs, StreamHandle);

int sdpa_run(const Tensor&, const Tensor&, const Tensor&, const Tensor*, Tensor&, const SDPAAttrs&, StreamHandle);
int sdpa_backward_run(const Tensor&, const Tensor&, const Tensor&, const Tensor*, const Tensor&, Tensor*, Tensor*, Tensor*, const SDPAAttrs&, StreamHandle);

int conv2d_run(const Tensor&, const Tensor&, const Tensor*, Tensor&, const Conv2DAttrs&, StreamHandle);
int conv2d_backward_run(const Tensor&, const Tensor&, const Tensor&, Tensor*, Tensor*, Tensor*, const Conv2DAttrs&, StreamHandle);

int maxpool2d_run(const Tensor&, Tensor&, Tensor*, const ai::Pool2DAttrs&, StreamHandle);
int maxpool2d_backward_run(const Tensor&, const Tensor&, Tensor&, const ai::Pool2DAttrs&, StreamHandle);

int avgpool2d_run(const Tensor&, Tensor&, const ai::Pool2DAttrs&, StreamHandle);
int avgpool2d_backward_run(const Tensor&, Tensor&, const ai::Pool2DAttrs&, StreamHandle);



}} // namespace ai::ops

// 헬퍼들
static inline ai::TensorDesc make_desc_4d_nchw(int64_t N,int64_t C,int64_t H,int64_t W){
  ai::TensorDesc d{}; d.dtype=ai::DType::F32; d.layout=ai::Layout::RowMajor;
  d.shape={N,C,H,W}; d.stride={C*H*W,H*W,W,1}; return d;
}
static inline ai::TensorDesc make_desc_1d(int64_t n){
  ai::TensorDesc d{}; d.dtype=ai::DType::F32; d.layout=ai::Layout::RowMajor;
  d.shape={n}; d.stride={1}; return d;
}

// -------------------- FWD: 단발 함수 --------------------
// ... (중략: GEMM, RMSNorm, LayerNorm, Softmax, CE, Dropout, Conv2D 등 기존 코드 변경 없음)
// 위 블록은 질문에 포함된 원본 그대로 유지하세요.

// ================== ⬇️⬇️⬇️ 여기부터 풀2D 바인딩 교체본 ⬇️⬇️⬇️ ==================

py::array gemm_bias_act(py::array A_in, py::array B_in,
                        py::object bias_in = py::none(),
                        std::string act = "relu",
                        double leaky_slope = 0.01)
{
  // NumPy → host f32 contiguous
  auto A_f = py::array_t<float, py::array::c_style | py::array::forcecast>(A_in);
  auto B_f = py::array_t<float, py::array::c_style | py::array::forcecast>(B_in);
  if (A_f.ndim()!=2 || B_f.ndim()!=2) throw std::runtime_error("A, B must be 2D");

  const int64_t M = A_f.shape(0);
  const int64_t K = A_f.shape(1);
  const int64_t Kb= B_f.shape(0);
  const int64_t N = B_f.shape(1);
  if (K != Kb) throw std::runtime_error("shape mismatch: A[M,K] @ B[K,N]");

  // Bias(optional)
  bool with_bias = false;
  py::array_t<float> Bias_f;
  int64_t bias_len = 0;
  if (!bias_in.is_none()) {
    Bias_f = py::array_t<float, py::array::c_style | py::array::forcecast>(bias_in);
    if (Bias_f.ndim() != 1) throw std::runtime_error("bias must be 1D (scalar/[M]/[N])");
    bias_len = Bias_f.shape(0);
    if (!(bias_len==1 || bias_len==M || bias_len==N)) {
      throw std::runtime_error("bias length must be 1, M or N");
    }
    with_bias = true;
  }

  // Device alloc/copy
  float *dA=nullptr, *dB=nullptr, *dBias=nullptr, *dY=nullptr;
  checkCuda(cudaMalloc(&dA, sizeof(float)*M*K), "cudaMalloc A");
  checkCuda(cudaMalloc(&dB, sizeof(float)*K*N), "cudaMalloc B");
  checkCuda(cudaMalloc(&dY, sizeof(float)*M*N), "cudaMalloc Y");
  checkCuda(cudaMemcpy(dA, A_f.data(), sizeof(float)*M*K, cudaMemcpyHostToDevice), "H2D A");
  checkCuda(cudaMemcpy(dB, B_f.data(), sizeof(float)*K*N, cudaMemcpyHostToDevice), "H2D B");
  if (with_bias) {
    checkCuda(cudaMalloc(&dBias, sizeof(float)*bias_len), "cudaMalloc Bias");
    checkCuda(cudaMemcpy(dBias, Bias_f.data(), sizeof(float)*bias_len, cudaMemcpyHostToDevice), "H2D Bias");
  }

  // Wrap tensors
  Tensor tA{dA, make_desc_2d(M,K), Device::CUDA, 0};
  Tensor tB{dB, make_desc_2d(K,N), Device::CUDA, 0};
  Tensor tY{dY, make_desc_2d(M,N), Device::CUDA, 0};

  Tensor tBias{}; Tensor* pBias=nullptr;
  if (with_bias) {
    TensorDesc bd{}; bd.dtype=DType::F32; bd.layout=Layout::RowMajor;
    if (bias_len==N)      bd.shape = {N};
    else if (bias_len==M) bd.shape = {M};
    else                  bd.shape = {1};
    bd.stride = {1};
    tBias = Tensor{dBias, bd, Device::CUDA, 0};
    pBias = &tBias;
  }

  GemmAttrs attrs{};
  attrs.act = parse_act(act);
  attrs.with_bias = with_bias;
  attrs.leaky_slope = static_cast<float>(leaky_slope);

  const int rc = ai::ops::gemm_run(tA, tB, pBias, tY, attrs, /*stream*/nullptr);
  checkCuda(cudaDeviceSynchronize(), "cuda sync after gemm");
  if (rc != 0) {
    cudaFree(dA); cudaFree(dB); cudaFree(dY); if (dBias) cudaFree(dBias);
    throw std::runtime_error("gemm_run failed with code " + std::to_string(rc));
  }

  // Result D2H
  auto Y_out = py::array_t<float>({M, N});
  checkCuda(cudaMemcpy(Y_out.mutable_data(), dY, sizeof(float)*M*N, cudaMemcpyDeviceToHost), "D2H Y");

  cudaFree(dA); cudaFree(dB); cudaFree(dY); if (dBias) cudaFree(dBias);
  return Y_out;
}

// -------------------- GemmPlan: 재사용/커널타이밍 --------------------
class GemmPlan {
public:
  GemmPlan(int64_t M, int64_t K, int64_t N,
           const std::string& act="relu",
           const std::string& bias_kind="pern",
           double leaky_slope=0.01)
    : M_(M), K_(K), N_(N)
  {
    if (M<=0||K<=0||N<=0) throw std::runtime_error("GemmPlan: invalid dims");
    attrs_.act = parse_act(act);
    attrs_.leaky_slope = static_cast<float>(leaky_slope);

    // bias kind: "none"|"pern"|"perm"|"scalar"
    std::string b; b.reserve(bias_kind.size());
    for (char c: bias_kind) b.push_back(std::tolower(static_cast<unsigned char>(c)));
    if (b=="none")      { bias_len_=0;  }
    else if (b=="pern") { bias_len_=N_; }
    else if (b=="perm") { bias_len_=M_; }
    else if (b=="scalar"){bias_len_=1;  }
    else throw std::runtime_error("GemmPlan: unknown bias_kind");

    attrs_.with_bias = (bias_len_>0);

    // device alloc once
    checkCuda(cudaMalloc(&dA_, sizeof(float)*M_*K_), "cudaMalloc dA");
    checkCuda(cudaMalloc(&dB_, sizeof(float)*K_*N_), "cudaMalloc dB");
    checkCuda(cudaMalloc(&dY_, sizeof(float)*M_*N_), "cudaMalloc dY");
    if (attrs_.with_bias) {
      checkCuda(cudaMalloc(&dBias_, sizeof(float)*bias_len_), "cudaMalloc dBias");
    }
  }

  ~GemmPlan() {
    if (dA_) cudaFree(dA_);
    if (dB_) cudaFree(dB_);
    if (dY_) cudaFree(dY_);
    if (dBias_) cudaFree(dBias_);
  }

  void upload(py::array A_in, py::array B_in, py::object bias_in = py::none()) {
    auto A = py::array_t<float, py::array::c_style | py::array::forcecast>(A_in);
    auto B = py::array_t<float, py::array::c_style | py::array::forcecast>(B_in);
    if (A.ndim()!=2 || B.ndim()!=2) throw std::runtime_error("upload: A,B must be 2D");
    if (A.shape(0)!=M_ || A.shape(1)!=K_) throw std::runtime_error("upload: A shape mismatch");
    if (B.shape(0)!=K_ || B.shape(1)!=N_) throw std::runtime_error("upload: B shape mismatch");

    checkCuda(cudaMemcpy(dA_, A.data(), sizeof(float)*M_*K_, cudaMemcpyHostToDevice), "H2D A");
    checkCuda(cudaMemcpy(dB_, B.data(), sizeof(float)*K_*N_, cudaMemcpyHostToDevice), "H2D B");

    if (attrs_.with_bias) {
      if (bias_in.is_none()) throw std::runtime_error("upload: bias_kind requires bias array");
      auto bias = py::array_t<float, py::array::c_style | py::array::forcecast>(bias_in);
      if (bias.ndim()!=1 || bias.shape(0)!=bias_len_) throw std::runtime_error("upload: bias length mismatch");
      checkCuda(cudaMemcpy(dBias_, bias.data(), sizeof(float)*bias_len_, cudaMemcpyHostToDevice), "H2D Bias");
    }
  }

  // 커널만 타이밍(ms) 반환. copy_out=True면 D2H까지 수행(시간에는 포함 안 함).
  float run(bool copy_out=false, py::object out_array = py::none()) {
    // wrap tensors
    Tensor tA{dA_, make_desc_2d(M_,K_), Device::CUDA, 0};
    Tensor tB{dB_, make_desc_2d(K_,N_), Device::CUDA, 0};
    Tensor tY{dY_, make_desc_2d(M_,N_), Device::CUDA, 0};
    Tensor tBias{}; Tensor* pBias=nullptr;
    if (attrs_.with_bias) {
      TensorDesc bd{}; bd.dtype=DType::F32; bd.layout=Layout::RowMajor;
      if (bias_len_==N_)      bd.shape = {N_};
      else if (bias_len_==M_) bd.shape = {M_};
      else                    bd.shape = {1};
      bd.stride = {1};
      tBias = Tensor{dBias_, bd, Device::CUDA, 0};
      pBias = &tBias;
    }

    // CUDA events for kernel timing
    cudaEvent_t ev0, ev1;
    checkCuda(cudaEventCreate(&ev0), "event create 0");
    checkCuda(cudaEventCreate(&ev1), "event create 1");

    checkCuda(cudaEventRecord(ev0, /*stream*/0), "event record 0");
    const int rc = ai::ops::gemm_run(tA, tB, pBias, tY, attrs_, /*stream*/nullptr);
    if (rc != 0) {
      cudaEventDestroy(ev0); cudaEventDestroy(ev1);
      throw std::runtime_error("gemm_run failed with code " + std::to_string(rc));
    }
    checkCuda(cudaEventRecord(ev1, /*stream*/0), "event record 1");
    checkCuda(cudaEventSynchronize(ev1), "event sync 1");

    float ms = 0.f;
    checkCuda(cudaEventElapsedTime(&ms, ev0, ev1), "event elapsed");
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);

    if (copy_out) {
      if (out_array.is_none()) {
        checkCuda(cudaDeviceSynchronize(), "sync after run(copy_out)");
      } else {
        auto Y = py::array_t<float>({M_, N_});
        checkCuda(cudaMemcpy(Y.mutable_data(), dY_, sizeof(float)*M_*N_, cudaMemcpyDeviceToHost), "D2H Y");
        // run()은 ms만 반환하는 API → 값은 반환하지 않음.
      }
    }
    return ms; // 커널 시간만(ms)
  }

  py::array get_output() const {
    auto Y = py::array_t<float>({M_, N_});
    checkCuda(cudaMemcpy(Y.mutable_data(), dY_, sizeof(float)*M_*N_, cudaMemcpyDeviceToHost), "D2H Y");
    return Y;
  }

  int64_t M() const { return M_; }
  int64_t K() const { return K_; }
  int64_t N() const { return N_; }
  int64_t bias_len() const { return bias_len_; }

private:
  int64_t M_{0}, K_{0}, N_{0};
  int64_t bias_len_{0};
  float *dA_{nullptr}, *dB_{nullptr}, *dY_{nullptr}, *dBias_{nullptr};
  GemmAttrs attrs_{};
};

// -------------------- FWD with Z: EX 경로 직접 호출 --------------------
static py::tuple gemm_bias_act_fwd_with_z(py::array A_in, py::array B_in,
                                          py::object bias_in = py::none(),
                                          std::string act = "relu",
                                          double leaky_slope = 0.01)
{
  auto A = py::array_t<float, py::array::c_style | py::array::forcecast>(A_in);
  auto B = py::array_t<float, py::array::c_style | py::array::forcecast>(B_in);
  if (A.ndim()!=2 || B.ndim()!=2) throw std::runtime_error("A,B must be 2D");
  int64_t M=A.shape(0), K=A.shape(1), Kb=B.shape(0), N=B.shape(1);
  if (K!=Kb) throw std::runtime_error("shape mismatch");

  // Bias(optional)
  float* dBias=nullptr; int64_t bias_len=0;
  regemm::BiasKind bk = regemm::BiasKind::None;
  if (!bias_in.is_none()) {
    auto bias = py::array_t<float, py::array::c_style | py::array::forcecast>(bias_in);
    if (bias.ndim()!=1) throw std::runtime_error("bias must be 1D");
    bias_len = bias.shape(0);
    if (bias_len==1)      bk = regemm::BiasKind::Scalar;
    else if (bias_len==M) bk = regemm::BiasKind::PerM;
    else if (bias_len==N) bk = regemm::BiasKind::PerN;
    else throw std::runtime_error("bias length must be 1|M|N");
    checkCuda(cudaMalloc(&dBias, sizeof(float)*bias_len), "cudaMalloc bias");
    checkCuda(cudaMemcpy(dBias, bias.data(), sizeof(float)*bias_len, cudaMemcpyHostToDevice), "H2D bias");
  }

  // Device buffers
  float *dA=nullptr,*dB=nullptr,*dY=nullptr,*dZ=nullptr;
  checkCuda(cudaMalloc(&dA, sizeof(float)*M*K), "cudaMalloc A");
  checkCuda(cudaMalloc(&dB, sizeof(float)*K*N), "cudaMalloc B");
  checkCuda(cudaMalloc(&dY, sizeof(float)*M*N), "cudaMalloc Y");
  checkCuda(cudaMalloc(&dZ, sizeof(float)*M*N), "cudaMalloc Z");
  checkCuda(cudaMemcpy(dA, A.data(), sizeof(float)*M*K, cudaMemcpyHostToDevice), "H2D A");
  checkCuda(cudaMemcpy(dB, B.data(), sizeof(float)*K*N, cudaMemcpyHostToDevice), "H2D B");

  // ParamsEx
  regemm::GemmBiasActParamsEx p{};
  p.M=(int)M; p.N=(int)N; p.K=(int)K;
  p.A=dA; p.lda=(int)K; p.B=dB; p.ldb=(int)N;
  p.C=nullptr; p.ldc=0;
  p.D=dY; p.ldd=(int)N;
  p.alpha=1.f; p.beta=0.f;
  p.bias=dBias; p.bias_kind=bk;
  p.act = to_regemm_act(parse_act(act));
  p.leaky_slope=(float)leaky_slope;
  p.Z=dZ; p.ldZ=(int)N; p.save_preact=1;

  // Launch
  regemm::gemm_bias_act_f32_ex(p, /*stream*/0);
  checkCuda(cudaDeviceSynchronize(), "sync after fwd_with_z");

  // D2H
  auto Y = py::array_t<float>({M,N});
  auto Z = py::array_t<float>({M,N});
  checkCuda(cudaMemcpy(Y.mutable_data(), dY, sizeof(float)*M*N, cudaMemcpyDeviceToHost), "D2H Y");
  checkCuda(cudaMemcpy(Z.mutable_data(), dZ, sizeof(float)*M*N, cudaMemcpyDeviceToHost), "D2H Z");

  // free
  if (dBias) cudaFree(dBias);
  cudaFree(dA); cudaFree(dB); cudaFree(dY); cudaFree(dZ);

  return py::make_tuple(Y, Z);
}

// -------------------- BWD: 디스패치 경유 --------------------
static py::dict gemm_bias_act_bwd(py::array A_in, py::array B_in,
                                  py::array gY_in, py::array Z_in,
                                  std::string act="relu",
                                  std::string bias_kind="none",
                                  double leaky_slope=0.01)
{
  auto A = py::array_t<float, py::array::c_style | py::array::forcecast>(A_in);
  auto B = py::array_t<float, py::array::c_style | py::array::forcecast>(B_in);
  auto gY= py::array_t<float, py::array::c_style | py::array::forcecast>(gY_in);
  auto Z = py::array_t<float, py::array::c_style | py::array::forcecast>(Z_in);
  if (A.ndim()!=2||B.ndim()!=2||gY.ndim()!=2||Z.ndim()!=2) throw std::runtime_error("all must be 2D");
  int64_t M=A.shape(0), K=A.shape(1), Kb=B.shape(0), N=B.shape(1);
  if (K!=Kb) throw std::runtime_error("shape mismatch A,B");
  if (gY.shape(0)!=M || gY.shape(1)!=N) throw std::runtime_error("gY shape mismatch");
  if (Z.shape(0)!=M  || Z.shape(1)!=N)  throw std::runtime_error("Z shape mismatch");

  // Device in
  float *dA=nullptr,*dB=nullptr,*dgY=nullptr,*dZ=nullptr;
  checkCuda(cudaMalloc(&dA, sizeof(float)*M*K), "cudaMalloc A");
  checkCuda(cudaMalloc(&dB, sizeof(float)*K*N), "cudaMalloc B");
  checkCuda(cudaMalloc(&dgY,sizeof(float)*M*N), "cudaMalloc gY");
  checkCuda(cudaMalloc(&dZ, sizeof(float)*M*N), "cudaMalloc Z");
  checkCuda(cudaMemcpy(dA, A.data(),  sizeof(float)*M*K, cudaMemcpyHostToDevice), "H2D A");
  checkCuda(cudaMemcpy(dB, B.data(),  sizeof(float)*K*N, cudaMemcpyHostToDevice), "H2D B");
  checkCuda(cudaMemcpy(dgY,gY.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice), "H2D gY");
  checkCuda(cudaMemcpy(dZ, Z.data(),  sizeof(float)*M*N, cudaMemcpyHostToDevice), "H2D Z");

  // Device outs
  float *dGA=nullptr,*dGB=nullptr,*dGC=nullptr,*dGBias=nullptr;
  checkCuda(cudaMalloc(&dGA, sizeof(float)*M*K), "cudaMalloc gA");
  checkCuda(cudaMalloc(&dGB, sizeof(float)*K*N), "cudaMalloc gB");

  // gBias shape 결정 (원래 FWD bias 축과 동일해야 함)
  int64_t gBias_len = 0;
  {
    std::string b; b.reserve(bias_kind.size());
    for (char c: bias_kind) b.push_back(std::tolower(static_cast<unsigned char>(c)));
    if      (b=="scalar") gBias_len = 1;
    else if (b=="perm")   gBias_len = M;
    else if (b=="pern")   gBias_len = N;
    else if (b=="none")   gBias_len = 0;
    else throw std::runtime_error("bias_kind must be one of: none|scalar|perm|pern");
  }
  if (gBias_len>0) {
    checkCuda(cudaMalloc(&dGBias, sizeof(float)*gBias_len), "cudaMalloc gBias");
  }

  // Wrap tensors
  auto d2 = make_desc_2d;
  Tensor tA{dA, d2(M,K), Device::CUDA, 0};
  Tensor tB{dB, d2(K,N), Device::CUDA, 0};
  Tensor tgY{dgY, d2(M,N), Device::CUDA, 0};
  Tensor tZ {dZ,  d2(M,N), Device::CUDA, 0};
  Tensor tGA{dGA, d2(M,K), Device::CUDA, 0};
  Tensor tGB{dGB, d2(K,N), Device::CUDA, 0};

  Tensor *pGC=nullptr, tGC{}; // C/gC 경로 미사용
  Tensor *pGBias=nullptr, tGBias{};
  if (gBias_len>0) {
    TensorDesc bd{}; bd.dtype=DType::F32; bd.layout=Layout::RowMajor; bd.stride={1};
    if      (gBias_len==1) bd.shape={1};
    else if (gBias_len==M) bd.shape={M};
    else                   bd.shape={N};
    tGBias = Tensor{dGBias, bd, Device::CUDA, 0};
    pGBias = &tGBias;
  }

  GemmAttrs at{}; at.act=parse_act(act); at.leaky_slope=(float)leaky_slope; at.with_bias=(gBias_len>0);

  int rc = ai::ops::gemm_bwd_run(tA, tB, /*C*/nullptr, tgY, tZ,
                                 &tGA, &tGB, /*gC*/pGC, /*gBias*/pGBias,
                                 at, /*stream*/nullptr);
  checkCuda(cudaDeviceSynchronize(), "sync after bwd");
  if (rc!=0) {
    if (dGBias) cudaFree(dGBias);
    cudaFree(dGA); cudaFree(dGB); cudaFree(dA); cudaFree(dB); cudaFree(dgY); cudaFree(dZ);
    throw std::runtime_error("gemm_bwd_run failed with code " + std::to_string(rc));
  }

  // D2H
  py::dict out;
  auto gA = py::array_t<float>({M,K});
  auto gB = py::array_t<float>({K,N});
  checkCuda(cudaMemcpy(gA.mutable_data(), dGA, sizeof(float)*M*K, cudaMemcpyDeviceToHost), "D2H gA");
  checkCuda(cudaMemcpy(gB.mutable_data(), dGB, sizeof(float)*K*N, cudaMemcpyDeviceToHost), "D2H gB");
  out["gA"] = gA; out["gB"] = gB;

  if (dGBias) {
    py::array_t<float> gBias({gBias_len});
    checkCuda(cudaMemcpy(gBias.mutable_data(), dGBias, sizeof(float)*gBias_len, cudaMemcpyDeviceToHost), "D2H gBias");
    out["gBias"] = gBias;
  } else {
    out["gBias"] = py::none();
  }
  out["gC"] = py::none(); // C 경로 미사용

  // free
  if (dGBias) cudaFree(dGBias);
  cudaFree(dGA); cudaFree(dGB); cudaFree(dA); cudaFree(dB); cudaFree(dgY); cudaFree(dZ);
  return out;
}

// py 함수 (전략: 기존 GEMM 바인딩 코드 패턴 재사용)
py::array rmsnorm(py::array X_in,
                  py::object gamma_in = py::none(),
                  py::object beta_in  = py::none(),
                  double eps = 1e-6)
{
  // --- 1) NumPy → host f32 contiguous ---
  auto X = py::array_t<float, py::array::c_style | py::array::forcecast>(X_in);
  if (X.ndim()!=2) throw std::runtime_error("rmsnorm: X must be 2D");
  const int64_t M = X.shape(0);
  const int64_t N = X.shape(1);

  // gamma / beta (optional, 1D length == N)
  bool has_gamma = !gamma_in.is_none();
  bool has_beta  = !beta_in.is_none();

  py::array_t<float> gamma_f, beta_f;
  if (has_gamma) {
    gamma_f = py::array_t<float, py::array::c_style | py::array::forcecast>(gamma_in);
    if (gamma_f.ndim()!=1 || gamma_f.shape(0)!=N)
      throw std::runtime_error("rmsnorm: gamma must be 1D with length N");
  }
  if (has_beta) {
    beta_f = py::array_t<float, py::array::c_style | py::array::forcecast>(beta_in);
    if (beta_f.ndim()!=1 || beta_f.shape(0)!=N)
      throw std::runtime_error("rmsnorm: beta must be 1D with length N");
  }

  // --- 2) Device alloc/copy ---
  float *dX=nullptr, *dGamma=nullptr, *dBeta=nullptr, *dY=nullptr;
  checkCuda(cudaMalloc(&dX,    sizeof(float)*M*N), "cudaMalloc X");
  checkCuda(cudaMalloc(&dY,    sizeof(float)*M*N), "cudaMalloc Y");
  checkCuda(cudaMemcpy(dX, X.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice), "H2D X");

  if (has_gamma) {
    checkCuda(cudaMalloc(&dGamma, sizeof(float)*N), "cudaMalloc gamma");
    checkCuda(cudaMemcpy(dGamma, gamma_f.data(), sizeof(float)*N, cudaMemcpyHostToDevice), "H2D gamma");
  }
  if (has_beta) {
    checkCuda(cudaMalloc(&dBeta, sizeof(float)*N), "cudaMalloc beta");
    checkCuda(cudaMemcpy(dBeta, beta_f.data(), sizeof(float)*N, cudaMemcpyHostToDevice), "H2D beta");
  }

  // --- 3) Wrap tensors ---
  auto make_desc_2d_local = [](int64_t r, int64_t c){
    TensorDesc d{}; d.dtype=DType::F32; d.layout=Layout::RowMajor; d.shape={r,c}; d.stride={c,1}; return d;
  };
  auto make_desc_1d = [](int64_t len){
    TensorDesc d{}; d.dtype=DType::F32; d.layout=Layout::RowMajor; d.shape={len}; d.stride={1}; return d;
  };

  Tensor tX{dX, make_desc_2d_local(M,N), Device::CUDA, 0};
  Tensor tY{dY, make_desc_2d_local(M,N), Device::CUDA, 0};

  Tensor tGamma{}, tBeta{};
  Tensor *pGamma = nullptr, *pBeta = nullptr;
  if (has_gamma) { tGamma = Tensor{dGamma, make_desc_1d(N), Device::CUDA, 0}; pGamma=&tGamma; }
  if (has_beta ) { tBeta  = Tensor{dBeta,  make_desc_1d(N), Device::CUDA, 0}; pBeta =&tBeta;  }

  // --- 4) Call op ---
  ai::RMSNormAttrs attrs{}; attrs.eps = static_cast<float>(eps);  // ✅ 올바른 타입/네임스페이스
  int rc = ai::ops::rmsnorm_run(tX, pGamma, pBeta, tY, attrs, /*stream*/nullptr);
  checkCuda(cudaDeviceSynchronize(), "sync after rmsnorm");
  if (rc != 0) {
    if (dGamma) cudaFree(dGamma);
    if (dBeta)  cudaFree(dBeta);
    cudaFree(dX); cudaFree(dY);
    throw std::runtime_error("rmsnorm_run failed with code " + std::to_string(rc));
  }

  // --- 5) D2H & free ---
  auto Y_out = py::array_t<float>({M, N});
  checkCuda(cudaMemcpy(Y_out.mutable_data(), dY, sizeof(float)*M*N, cudaMemcpyDeviceToHost), "D2H Y");

  if (dGamma) cudaFree(dGamma);
  if (dBeta)  cudaFree(dBeta);
  cudaFree(dX); cudaFree(dY);
  return Y_out;
}

py::tuple rmsnorm_backward(py::array X_in,
                           py::object gamma_in,
                           py::array dY_in,
                           double eps = 1e-6)
{
  // --- 1) NumPy → host f32 contiguous ---
  auto X  = py::array_t<float, py::array::c_style | py::array::forcecast>(X_in);
  auto dY = py::array_t<float, py::array::c_style | py::array::forcecast>(dY_in);
  if (X.ndim()!=2 || dY.ndim()!=2) throw std::runtime_error("rmsnorm_backward: X, dY must be 2D");

  const int64_t M = X.shape(0);
  const int64_t N = X.shape(1);
  if (dY.shape(0)!=M || dY.shape(1)!=N) throw std::runtime_error("rmsnorm_backward: dY shape mismatch");

  bool has_gamma = !gamma_in.is_none();
  py::array_t<float> gamma_f;
  if (has_gamma) {
    gamma_f = py::array_t<float, py::array::c_style | py::array::forcecast>(gamma_in);
    if (gamma_f.ndim()!=1 || gamma_f.shape(0)!=N)
      throw std::runtime_error("rmsnorm_backward: gamma must be 1D with length N");
  }

  // --- 2) Device alloc/copy ---
  float *dX=nullptr, *dGamma=nullptr, *dYdev=nullptr;
  float *dDX=nullptr, *dDGamma=nullptr, *dDBeta=nullptr;

  checkCuda(cudaMalloc(&dX,     sizeof(float)*M*N), "cudaMalloc X");
  checkCuda(cudaMalloc(&dYdev,  sizeof(float)*M*N), "cudaMalloc dY");
  checkCuda(cudaMalloc(&dDX,    sizeof(float)*M*N), "cudaMalloc dX");
  checkCuda(cudaMemcpy(dX,    X.data(),  sizeof(float)*M*N, cudaMemcpyHostToDevice), "H2D X");
  checkCuda(cudaMemcpy(dYdev, dY.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice), "H2D dY");

  if (has_gamma) {
    checkCuda(cudaMalloc(&dGamma,  sizeof(float)*N), "cudaMalloc gamma");
    checkCuda(cudaMalloc(&dDGamma, sizeof(float)*N), "cudaMalloc dgamma");
    checkCuda(cudaMemcpy(dGamma, gamma_f.data(), sizeof(float)*N, cudaMemcpyHostToDevice), "H2D gamma");
    checkCuda(cudaMemset(dDGamma, 0, sizeof(float)*N), "memset dgamma");
  }
  // dbeta는 러닝 합만 필요하므로 길이 N. (Affine 사용 시에만 계산/반환)
  if (has_gamma) {
    checkCuda(cudaMalloc(&dDBeta, sizeof(float)*N), "cudaMalloc dbeta");
    checkCuda(cudaMemset(dDBeta, 0, sizeof(float)*N), "memset dbeta");
  }

  // --- 3) Wrap tensors ---
  auto make_desc_2d_local = [](int64_t r, int64_t c){
    TensorDesc d{}; d.dtype=DType::F32; d.layout=Layout::RowMajor; d.shape={r,c}; d.stride={c,1}; return d;
  };
  auto make_desc_1d = [](int64_t len){
    TensorDesc d{}; d.dtype=DType::F32; d.layout=Layout::RowMajor; d.shape={len}; d.stride={1}; return d;
  };

  Tensor tX {dX,    make_desc_2d_local(M,N), Device::CUDA, 0};
  Tensor tdY{dYdev, make_desc_2d_local(M,N), Device::CUDA, 0};
  Tensor tdX{dDX,   make_desc_2d_local(M,N), Device::CUDA, 0};

  Tensor tGamma{}, tDGamma{}, tDBeta{};
  Tensor *pGamma = nullptr, *pDGamma = nullptr, *pDBeta = nullptr;

  if (has_gamma) {
    tGamma  = Tensor{dGamma,  make_desc_1d(N), Device::CUDA, 0}; pGamma  = &tGamma;
    tDGamma = Tensor{dDGamma, make_desc_1d(N), Device::CUDA, 0}; pDGamma = &tDGamma;
    tDBeta  = Tensor{dDBeta,  make_desc_1d(N), Device::CUDA, 0}; pDBeta  = &tDBeta;
  }

  // --- 4) Call op ---
  ai::RMSNormAttrs attrs{}; attrs.eps = static_cast<float>(eps);  // ✅ 올바른 타입/네임스페이스
  int rc = ai::ops::rmsnorm_backward_run(tX, pGamma, tdY, tdX, pDGamma, pDBeta, attrs, /*stream*/nullptr);
  checkCuda(cudaDeviceSynchronize(), "sync after rmsnorm_backward");
  if (rc != 0) {
    if (dDGamma) cudaFree(dDGamma);
    if (dDBeta)  cudaFree(dDBeta);
    if (dGamma)  cudaFree(dGamma);
    cudaFree(dDX); cudaFree(dX); cudaFree(dYdev);
    throw std::runtime_error("rmsnorm_backward_run failed with code " + std::to_string(rc));
  }

  // --- 5) D2H ---
  auto dX_out = py::array_t<float>({M, N});
  checkCuda(cudaMemcpy(dX_out.mutable_data(), dDX, sizeof(float)*M*N, cudaMemcpyDeviceToHost), "D2H dX");

  py::object dgamma_obj = py::none();
  py::object dbeta_obj  = py::none();

  if (has_gamma) {
    auto dgamma_out = py::array_t<float>({N});
    auto dbeta_out  = py::array_t<float>({N});
    checkCuda(cudaMemcpy(dgamma_out.mutable_data(), dDGamma, sizeof(float)*N, cudaMemcpyDeviceToHost), "D2H dgamma");
    checkCuda(cudaMemcpy(dbeta_out.mutable_data(),  dDBeta,  sizeof(float)*N, cudaMemcpyDeviceToHost),  "D2H dbeta");
    dgamma_obj = std::move(dgamma_out);
    dbeta_obj  = std::move(dbeta_out);
  }

  // --- 6) free & return ---
  if (dDGamma) cudaFree(dDGamma);
  if (dDBeta)  cudaFree(dDBeta);
  if (dGamma)  cudaFree(dGamma);
  cudaFree(dDX); cudaFree(dX); cudaFree(dYdev);

  return py::make_tuple(std::move(dX_out), dgamma_obj, dbeta_obj);
}
static inline ai::TensorDesc make_desc_2d_ln(int64_t r, int64_t c){
  ai::TensorDesc d{}; d.dtype=ai::DType::F32; d.layout=ai::Layout::RowMajor; d.shape={r,c}; d.stride={c,1}; return d;
}
static inline ai::TensorDesc make_desc_1d_ln(int64_t n){
  ai::TensorDesc d{}; d.dtype=ai::DType::F32; d.layout=ai::Layout::RowMajor; d.shape={n}; d.stride={1}; return d;
}

py::array layernorm(py::array X_in,
                    py::object gamma_in = py::none(),
                    py::object beta_in  = py::none(),
                    double eps = 1e-5)
{
  auto X = py::array_t<float, py::array::c_style | py::array::forcecast>(X_in);
  if (X.ndim()!=2) throw std::runtime_error("layernorm: X must be 2D");
  const int64_t M = X.shape(0), N = X.shape(1);

  bool has_gamma = !gamma_in.is_none();
  bool has_beta  = !beta_in.is_none();
  py::array_t<float> gamma_f, beta_f;
  if (has_gamma) { gamma_f = py::array_t<float, py::array::c_style | py::array::forcecast>(gamma_in);
                   if (gamma_f.ndim()!=1 || gamma_f.shape(0)!=N) throw std::runtime_error("gamma must be 1D len N"); }
  if (has_beta)  { beta_f  = py::array_t<float, py::array::c_style | py::array::forcecast>(beta_in);
                   if (beta_f.ndim()!=1 || beta_f.shape(0)!=N) throw std::runtime_error("beta must be 1D len N"); }

  float *dX=nullptr,*dY=nullptr,*dG=nullptr,*dB=nullptr;
  cudaMalloc(&dX, sizeof(float)*M*N);
  cudaMalloc(&dY, sizeof(float)*M*N);
  cudaMemcpy(dX, X.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice);
  if (has_gamma) { cudaMalloc(&dG, sizeof(float)*N); cudaMemcpy(dG, gamma_f.data(), sizeof(float)*N, cudaMemcpyHostToDevice); }
  if (has_beta ) { cudaMalloc(&dB, sizeof(float)*N); cudaMemcpy(dB,  beta_f.data(), sizeof(float)*N,  cudaMemcpyHostToDevice); }

  ai::Tensor tX{dX, make_desc_2d_ln(M,N), ai::Device::CUDA, 0};
  ai::Tensor tY{dY, make_desc_2d_ln(M,N), ai::Device::CUDA, 0};
  ai::Tensor tG{}, tB{};
  ai::Tensor *pG=nullptr, *pB=nullptr;
  if (has_gamma) { tG={dG, make_desc_1d_ln(N), ai::Device::CUDA, 0}; pG=&tG; }
  if (has_beta ) { tB={dB, make_desc_1d_ln(N), ai::Device::CUDA, 0}; pB=&tB; }

  ai::LayerNormAttrs attrs{}; attrs.eps=(float)eps;
  int rc = ai::ops::layernorm_run(tX, pG, pB, tY, attrs, nullptr);
  cudaDeviceSynchronize();
  if (rc!=0) {
    if (dG) cudaFree(dG); if (dB) cudaFree(dB);
    cudaFree(dX); cudaFree(dY);
    throw std::runtime_error("layernorm_run failed: " + std::to_string(rc));
  }

  auto Y = py::array_t<float>({M,N});
  cudaMemcpy(Y.mutable_data(), dY, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
  if (dG) cudaFree(dG); if (dB) cudaFree(dB);
  cudaFree(dX); cudaFree(dY);
  return Y;
}

py::tuple layernorm_backward(py::array X_in,
                             py::object gamma_in,
                             py::array dY_in,
                             double eps = 1e-5)
{
  auto X  = py::array_t<float, py::array::c_style | py::array::forcecast>(X_in);
  auto dY = py::array_t<float, py::array::c_style | py::array::forcecast>(dY_in);
  if (X.ndim()!=2 || dY.ndim()!=2) throw std::runtime_error("layernorm_backward: X,dY must be 2D");
  const int64_t M=X.shape(0), N=X.shape(1);
  if (dY.shape(0)!=M || dY.shape(1)!=N) throw std::runtime_error("dY shape mismatch");

  bool has_gamma = !gamma_in.is_none();
  py::array_t<float> gamma_f;
  if (has_gamma) { gamma_f = py::array_t<float, py::array::c_style | py::array::forcecast>(gamma_in);
                   if (gamma_f.ndim()!=1 || gamma_f.shape(0)!=N) throw std::runtime_error("gamma must be 1D len N"); }

  float *dX=nullptr,*dYd=nullptr,*dG=nullptr,*dDX=nullptr,*dDG=nullptr,*dDB=nullptr;
  cudaMalloc(&dX,  sizeof(float)*M*N);
  cudaMalloc(&dYd, sizeof(float)*M*N);
  cudaMalloc(&dDX, sizeof(float)*M*N);
  cudaMemcpy(dX,  X.data(),  sizeof(float)*M*N, cudaMemcpyHostToDevice);
  cudaMemcpy(dYd, dY.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice);
  if (has_gamma) {
    cudaMalloc(&dG,  sizeof(float)*N);
    cudaMalloc(&dDG, sizeof(float)*N);
    cudaMalloc(&dDB, sizeof(float)*N);
    cudaMemcpy(dG, gamma_f.data(), sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemset(dDG, 0, sizeof(float)*N);
    cudaMemset(dDB, 0, sizeof(float)*N);
  }

  ai::Tensor tX{dX, make_desc_2d_ln(M,N), ai::Device::CUDA, 0};
  ai::Tensor tdY{dYd, make_desc_2d_ln(M,N), ai::Device::CUDA, 0};
  ai::Tensor tdX{dDX, make_desc_2d_ln(M,N), ai::Device::CUDA, 0};
  ai::Tensor tG{}, tDG{}, tDB{};
  ai::Tensor *pG=nullptr, *pDG=nullptr, *pDB=nullptr;
  if (has_gamma) {
    tG ={dG,  make_desc_1d_ln(N), ai::Device::CUDA, 0}; pG =&tG;
    tDG={dDG, make_desc_1d_ln(N), ai::Device::CUDA, 0}; pDG=&tDG;
    tDB={dDB, make_desc_1d_ln(N), ai::Device::CUDA, 0}; pDB=&tDB;
  }

  ai::LayerNormAttrs attrs{}; attrs.eps=(float)eps;
  int rc = ai::ops::layernorm_backward_run(tX, pG, tdY, tdX, pDG, pDB, attrs, nullptr);
  cudaDeviceSynchronize();
  if (rc!=0) {
    if (dDG) cudaFree(dDG); if (dDB) cudaFree(dDB); if (dG) cudaFree(dG);
    cudaFree(dDX); cudaFree(dX); cudaFree(dYd);
    throw std::runtime_error("layernorm_backward_run failed: " + std::to_string(rc));
  }

  auto dX_out = py::array_t<float>({M,N});
  cudaMemcpy(dX_out.mutable_data(), dDX, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

  py::object dgamma_obj = py::none(), dbeta_obj = py::none();
  if (has_gamma) {
    auto dgamma = py::array_t<float>({N});
    auto dbeta  = py::array_t<float>({N});
    cudaMemcpy(dgamma.mutable_data(), dDG, sizeof(float)*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(dbeta.mutable_data(),  dDB, sizeof(float)*N, cudaMemcpyDeviceToHost);
    dgamma_obj = std::move(dgamma);
    dbeta_obj  = std::move(dbeta);
  }

  if (dDG) cudaFree(dDG); if (dDB) cudaFree(dDB); if (dG) cudaFree(dG);
  cudaFree(dDX); cudaFree(dX); cudaFree(dYd);
  return py::make_tuple(std::move(dX_out), dgamma_obj, dbeta_obj);
}

// 바인딩 함수
py::array softmax(py::array X_in, py::object mask_in = py::none(),
                  double scale=1.0, bool log=false)
{
  auto X = py::array_t<float, py::array::c_style | py::array::forcecast>(X_in);
  if (X.ndim()!=2) throw std::runtime_error("softmax: X must be 2D");
  int64_t M=X.shape(0), N=X.shape(1);

  // device alloc
  float *dX=nullptr,*dY=nullptr,*dMask=nullptr;
  cudaMalloc(&dX, sizeof(float)*M*N);
  cudaMalloc(&dY, sizeof(float)*M*N);
  cudaMemcpy(dX, X.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice);

  Tensor tX{dX, make_desc_2d(M,N), Device::CUDA, 0};
  Tensor tY{dY, make_desc_2d(M,N), Device::CUDA, 0};

  Tensor tMask{}; Tensor* pMask=nullptr;
  if (!mask_in.is_none()) {
    auto Mk = py::array_t<float, py::array::c_style | py::array::forcecast>(mask_in);
    if (!((Mk.ndim()==2 && Mk.shape(0)==M && Mk.shape(1)==N) || (Mk.ndim()==1 && Mk.shape(0)==N)))
      throw std::runtime_error("mask must be [M,N] or [N]");
    size_t sz = (Mk.ndim()==2) ? (size_t)M*N : (size_t)N;
    cudaMalloc(&dMask, sizeof(float)*sz);
    cudaMemcpy(dMask, Mk.data(), sizeof(float)*sz, cudaMemcpyHostToDevice);
    TensorDesc md{}; md.dtype=DType::F32; md.layout=Layout::RowMajor; md.stride={ (int64_t)(Mk.ndim()==2?N:1), 1 };
    md.shape = (Mk.ndim()==2) ? std::vector<int64_t>{M,N} : std::vector<int64_t>{N};
    tMask = Tensor{dMask, md, Device::CUDA, 0};
    pMask = &tMask;
  }

  ai::SoftmaxAttrs attrs{}; attrs.scale=(float)scale; attrs.log=log;
  int rc = ai::ops::softmax_run(tX, pMask, tY, attrs, /*stream*/nullptr);
  cudaDeviceSynchronize();
  if (rc!=0) {
    if (dMask) cudaFree(dMask); cudaFree(dX); cudaFree(dY);
    throw std::runtime_error("softmax_run failed: " + std::to_string(rc));
  }

  auto Y = py::array_t<float>({M,N});
  cudaMemcpy(Y.mutable_data(), dY, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
  if (dMask) cudaFree(dMask); cudaFree(dX); cudaFree(dY);
  return Y;
}

py::array softmax_backward(py::array Y_in, py::array dY_in, bool log=false)
{
  auto Y  = py::array_t<float, py::array::c_style | py::array::forcecast>(Y_in);
  auto dY = py::array_t<float, py::array::c_style | py::array::forcecast>(dY_in);
  if (Y.ndim()!=2 || dY.ndim()!=2) throw std::runtime_error("softmax_backward: Y,dY must be 2D");
  if (Y.shape(0)!=dY.shape(0) || Y.shape(1)!=dY.shape(1)) throw std::runtime_error("shape mismatch");

  int64_t M=Y.shape(0), N=Y.shape(1);
  float *dYd=nullptr,*dYdev=nullptr,*dX=nullptr;
  cudaMalloc(&dYd,   sizeof(float)*M*N);
  cudaMalloc(&dYdev, sizeof(float)*M*N);
  cudaMemcpy(dYd, Y.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice);
  cudaMemcpy(dYdev, dY.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice);
  cudaMalloc(&dX, sizeof(float)*M*N);

  Tensor tY{dYd,   make_desc_2d(M,N), Device::CUDA, 0};
  Tensor tdY{dYdev,make_desc_2d(M,N), Device::CUDA, 0};
  Tensor tdX{dX,   make_desc_2d(M,N), Device::CUDA, 0};

  ai::SoftmaxAttrs attrs{}; attrs.scale=1.f; attrs.log=log;
  int rc = ai::ops::softmax_backward_run(tY, tdY, tdX, attrs, /*stream*/nullptr);
  cudaDeviceSynchronize();
  if (rc!=0) {
    cudaFree(dYd); cudaFree(dYdev); cudaFree(dX);
    throw std::runtime_error("softmax_backward_run failed: " + std::to_string(rc));
  }

  auto dX_out = py::array_t<float>({M,N});
  cudaMemcpy(dX_out.mutable_data(), dX, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
  cudaFree(dYd); cudaFree(dYdev); cudaFree(dX);
  return dX_out;
}

// ---- Cross Entropy: Forward ----
py::object cross_entropy(py::array X_in, py::array target_in,
                         std::string reduction = "mean",
                         int ignore_index = -1,
                         double label_smoothing = 0.0)
{
  auto X = py::array_t<float,   py::array::c_style | py::array::forcecast>(X_in);
  if (X.ndim()!=2) throw std::runtime_error("cross_entropy: X must be 2D");
  const int64_t M = X.shape(0), N = X.shape(1);

  auto T = py::array_t<int32_t, py::array::c_style | py::array::forcecast>(target_in);
  if (T.ndim()!=1 || T.shape(0)!=M) throw std::runtime_error("cross_entropy: target must be 1D len M");

  float   *dX=nullptr, *dL=nullptr;
  int32_t *dT=nullptr;
  cudaMalloc(&dX, sizeof(float)*M*N);
  cudaMalloc(&dT, sizeof(int32_t)*M);
  cudaMemcpy(dX, X.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice);
  cudaMemcpy(dT, T.data(), sizeof(int32_t)*M, cudaMemcpyHostToDevice);

  ai::Tensor tX{dX, make_desc_2d_any(M,N, ai::DType::F32), ai::Device::CUDA, 0};
  ai::Tensor tT{dT, make_desc_1d_any(M,   ai::DType::I32), ai::Device::CUDA, 0};

  ai::CrossEntropyAttrs attrs{};
  if      (reduction=="none") attrs.reduction = ai::Reduction::None;
  else if (reduction=="sum")  attrs.reduction = ai::Reduction::Sum;
  else                        attrs.reduction = ai::Reduction::Mean;
  attrs.ignore_index = ignore_index;
  attrs.ls_eps       = (float)label_smoothing;

  py::object out;
  if (attrs.reduction==ai::Reduction::None) {
    cudaMalloc(&dL, sizeof(float)*M);
    ai::Tensor tL{dL, make_desc_1d_any(M, ai::DType::F32), ai::Device::CUDA, 0};
    int rc = ai::ops::cross_entropy_run(tX, tT, tL, attrs, nullptr);
    cudaDeviceSynchronize();
    if (rc!=0) { cudaFree(dX); cudaFree(dT); cudaFree(dL); throw std::runtime_error("cross_entropy_run failed"); }
    auto L = py::array_t<float>({M});
    cudaMemcpy(L.mutable_data(), dL, sizeof(float)*M, cudaMemcpyDeviceToHost);
    out = std::move(L);
  } else {
    cudaMalloc(&dL, sizeof(float));
    ai::Tensor tL{dL, make_desc_1d_any(1, ai::DType::F32), ai::Device::CUDA, 0};
    int rc = ai::ops::cross_entropy_run(tX, tT, tL, attrs, nullptr);
    cudaDeviceSynchronize();
    if (rc!=0) { cudaFree(dX); cudaFree(dT); cudaFree(dL); throw std::runtime_error("cross_entropy_run failed"); }
    float host;
    cudaMemcpy(&host, dL, sizeof(float), cudaMemcpyDeviceToHost);
    out = py::float_(host);
  }

  cudaFree(dX); cudaFree(dT); cudaFree(dL);
  return out;
}

// ---- Cross Entropy: Backward ----
py::array cross_entropy_backward(py::array X_in, py::array target_in,
                                 std::string reduction = "mean",
                                 int ignore_index = -1,
                                 double label_smoothing = 0.0)
{
  auto X = py::array_t<float,   py::array::c_style | py::array::forcecast>(X_in);
  if (X.ndim()!=2) throw std::runtime_error("cross_entropy_backward: X must be 2D");
  const int64_t M = X.shape(0), N = X.shape(1);

  auto T = py::array_t<int32_t, py::array::c_style | py::array::forcecast>(target_in);
  if (T.ndim()!=1 || T.shape(0)!=M) throw std::runtime_error("cross_entropy_backward: target mismatch");

  float   *dX=nullptr, *dDX=nullptr;
  int32_t *dT=nullptr;
  cudaMalloc(&dX,  sizeof(float)*M*N);
  cudaMalloc(&dDX, sizeof(float)*M*N);
  cudaMalloc(&dT,  sizeof(int32_t)*M);
  cudaMemcpy(dX, X.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice);
  cudaMemcpy(dT, T.data(), sizeof(int32_t)*M, cudaMemcpyHostToDevice);

  ai::Tensor tX {dX,  make_desc_2d_any(M,N, ai::DType::F32), ai::Device::CUDA, 0};
  ai::Tensor tdX{dDX, make_desc_2d_any(M,N, ai::DType::F32), ai::Device::CUDA, 0};
  ai::Tensor tT {dT,  make_desc_1d_any(M,   ai::DType::I32), ai::Device::CUDA, 0};

  ai::CrossEntropyAttrs attrs{};
  if      (reduction=="none") attrs.reduction = ai::Reduction::None;
  else if (reduction=="sum")  attrs.reduction = ai::Reduction::Sum;
  else                        attrs.reduction = ai::Reduction::Mean;
  attrs.ignore_index = ignore_index;
  attrs.ls_eps       = (float)label_smoothing;

  int rc = ai::ops::cross_entropy_backward_run(tX, tT, tdX, attrs, nullptr);
  cudaDeviceSynchronize();
  if (rc!=0) { cudaFree(dX); cudaFree(dDX); cudaFree(dT); throw std::runtime_error("cross_entropy_backward_run failed"); }

  auto out = py::array_t<float>({M,N});
  cudaMemcpy(out.mutable_data(), dDX, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
  cudaFree(dX); cudaFree(dDX); cudaFree(dT);
  return out;
}

py::object dropout(py::array X_in,
                   double p = 0.1,
                   py::object return_mask = py::bool_(false),
                   uint64_t seed = 0x1234,
                   bool scale_in_train = true)
{
  auto X = py::array_t<float, py::array::c_style | py::array::forcecast>(X_in);
  if (X.ndim()!=2) throw std::runtime_error("dropout: X must be 2D");
  int64_t M=X.shape(0), N=X.shape(1);

  float *dX=nullptr, *dY=nullptr; int32_t* dM=nullptr;
  cudaMalloc(&dX, sizeof(float)*M*N);
  cudaMalloc(&dY, sizeof(float)*M*N);
  cudaMemcpy(dX, X.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice);
  if (py::cast<bool>(return_mask)) {
    cudaMalloc(&dM, sizeof(int32_t)*M*N);
  }

  ai::Tensor tX{dX, make_desc_2d_any(M,N, ai::DType::F32), ai::Device::CUDA, 0};
  ai::Tensor tY{dY, make_desc_2d_any(M,N, ai::DType::F32), ai::Device::CUDA, 0};
  ai::Tensor tM{}; ai::Tensor* pM=nullptr;
  if (dM){ tM = ai::Tensor{dM, make_desc_2d_any(M,N, ai::DType::I32), ai::Device::CUDA, 0}; pM=&tM; }

  ai::DropoutAttrs attrs{}; attrs.p=(float)p; attrs.seed=seed; attrs.scale_in_train=scale_in_train;
  int rc = ai::ops::dropout_run(tX, tY, pM, attrs, nullptr);
  cudaDeviceSynchronize();
  if (rc!=0){ if (dM) cudaFree(dM); cudaFree(dX); cudaFree(dY); throw std::runtime_error("dropout_run failed"); }

  auto Y = py::array_t<float>({M,N});
  cudaMemcpy(Y.mutable_data(), dY, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

  if (dM){
    auto Mhost = py::array_t<int32_t>({M,N});
    cudaMemcpy(Mhost.mutable_data(), dM, sizeof(int32_t)*M*N, cudaMemcpyDeviceToHost);
    cudaFree(dM); cudaFree(dX); cudaFree(dY);
    return py::make_tuple(std::move(Y), std::move(Mhost));
  } else {
    cudaFree(dX); cudaFree(dY);
    return Y;
  }
}

py::array dropout_backward(py::array dY_in, py::array mask_in,
                           double p = 0.1, uint64_t seed = 0x1234,
                           bool scale_in_train = true)
{
  auto dY = py::array_t<float,   py::array::c_style | py::array::forcecast>(dY_in);
  auto M  = py::array_t<int32_t, py::array::c_style | py::array::forcecast>(mask_in);
  if (dY.ndim()!=2 || M.ndim()!=2) throw std::runtime_error("dropout_backward: dY, mask must be 2D");
  if (dY.shape(0)!=M.shape(0) || dY.shape(1)!=M.shape(1)) throw std::runtime_error("shape mismatch");

  int64_t R=dY.shape(0), C=dY.shape(1);
  float *ddY=nullptr, *ddX=nullptr; int32_t* dM=nullptr;
  cudaMalloc(&ddY, sizeof(float)*R*C);
  cudaMalloc(&ddX, sizeof(float)*R*C);
  cudaMalloc(&dM,  sizeof(int32_t)*R*C);
  cudaMemcpy(ddY, dY.data(), sizeof(float)*R*C, cudaMemcpyHostToDevice);
  cudaMemcpy(dM,  M.data(),  sizeof(int32_t)*R*C, cudaMemcpyHostToDevice);

  ai::Tensor tdY{ddY, make_desc_2d_any(R,C, ai::DType::F32), ai::Device::CUDA, 0};
  ai::Tensor tM {dM,  make_desc_2d_any(R,C, ai::DType::I32), ai::Device::CUDA, 0};
  ai::Tensor tdX{ddX, make_desc_2d_any(R,C, ai::DType::F32), ai::Device::CUDA, 0};

  ai::DropoutAttrs attrs{}; attrs.p=(float)p; attrs.seed=seed; attrs.scale_in_train=scale_in_train;
  int rc = ai::ops::dropout_backward_run(tdY, tM, tdX, attrs, nullptr);
  cudaDeviceSynchronize();
  if (rc!=0){ cudaFree(ddY); cudaFree(ddX); cudaFree(dM); throw std::runtime_error("dropout_backward_run failed"); }

  auto out = py::array_t<float>({R,C});
  cudaMemcpy(out.mutable_data(), ddX, sizeof(float)*R*C, cudaMemcpyDeviceToHost);
  cudaFree(ddY); cudaFree(ddX); cudaFree(dM);
  return out;
}

// ---- FWD ----
py::array conv2d(py::array X_in, py::array W_in, py::object B_in = py::none(),
                 int stride_h=1,int stride_w=1,int pad_h=0,int pad_w=0,
                 int dil_h=1,int dil_w=1)
{
  auto X = py::array_t<float, py::array::c_style | py::array::forcecast>(X_in);
  auto W = py::array_t<float, py::array::c_style | py::array::forcecast>(W_in);
  if (X.ndim()!=4 || W.ndim()!=4) throw std::runtime_error("conv2d: X[N,C,H,W], W[Cout,Cin,Kh,Kw] required");

  const int64_t N=X.shape(0), Cin=X.shape(1), H=X.shape(2), Ww=X.shape(3);
  const int64_t Cout=W.shape(0), WCin=W.shape(1), Kh=W.shape(2), Kw=W.shape(3);
  if (WCin!=Cin) throw std::runtime_error("conv2d: Cin mismatch between X and W");

  const int64_t Ho = (H + 2*pad_h - dil_h*(Kh-1) - 1)/stride_h + 1;
  const int64_t Wo = (Ww+ 2*pad_w - dil_w*(Kw-1) - 1)/stride_w + 1;
  if (Ho<=0 || Wo<=0) throw std::runtime_error("conv2d: invalid output size (check stride/pad/dilation/KhKw)");

  float *dX=nullptr,*dW=nullptr,*dY=nullptr,*dB=nullptr;
  cudaMalloc(&dX, sizeof(float)*N*Cin*H*Ww);
  cudaMalloc(&dW, sizeof(float)*Cout*Cin*Kh*Kw);
  cudaMalloc(&dY, sizeof(float)*N*Cout*Ho*Wo);
  cudaMemcpy(dX, X.data(), sizeof(float)*N*Cin*H*Ww, cudaMemcpyHostToDevice);
  cudaMemcpy(dW, W.data(), sizeof(float)*Cout*Cin*Kh*Kw, cudaMemcpyHostToDevice);

  ai::Tensor tX{dX, make_desc_4d_nchw(N,Cin,H,Ww), ai::Device::CUDA, 0};
  ai::Tensor tW{dW, {ai::DType::F32, ai::Layout::RowMajor, {Cout,Cin,Kh,Kw},
                     {Cin*Kh*Kw, Kh*Kw, Kw, 1}}, ai::Device::CUDA, 0};
  ai::Tensor tY{dY, make_desc_4d_nchw(N,Cout,Ho,Wo), ai::Device::CUDA, 0};

  ai::Tensor tB{}, *pB=nullptr;
  if (!B_in.is_none()) {
    auto B = py::array_t<float, py::array::c_style | py::array::forcecast>(B_in);
    if (B.ndim()!=1 || B.shape(0)!=Cout) {
      cudaFree(dX); cudaFree(dW); cudaFree(dY);
      throw std::runtime_error("conv2d: bias must be 1D [Cout]");
    }
    cudaMalloc(&dB, sizeof(float)*Cout);
    cudaMemcpy(dB, B.data(), sizeof(float)*Cout, cudaMemcpyHostToDevice);
    tB = {dB, make_desc_1d_f32(Cout), ai::Device::CUDA, 0}; pB=&tB;
  }

  ai::Conv2DAttrs attrs{};
  attrs.stride_h = stride_h; attrs.stride_w = stride_w;
  attrs.pad_h = pad_h;       attrs.pad_w = pad_w;
  attrs.dil_h = dil_h;       attrs.dil_w = dil_w;
  attrs.groups = 1;

  const int rc = ai::ops::conv2d_run(tX, tW, pB, tY, attrs, /*stream*/nullptr);
  cudaDeviceSynchronize();
  if (rc!=0) {
    if (dB) cudaFree(dB); cudaFree(dX); cudaFree(dW); cudaFree(dY);
    throw std::runtime_error("conv2d_run failed with code " + std::to_string(rc));
  }

  auto Y = py::array_t<float>({N, Cout, Ho, Wo});
  cudaMemcpy(Y.mutable_data(), dY, sizeof(float)*N*Cout*Ho*Wo, cudaMemcpyDeviceToHost);
  if (dB) cudaFree(dB); cudaFree(dX); cudaFree(dW); cudaFree(dY);
  return Y;
}

// ---- BWD ----
py::tuple conv2d_backward(py::array X_in, py::array W_in, py::array dY_in,
                          bool need_dW=true, bool need_dB=true, bool need_dX=true,
                          int stride_h=1,int stride_w=1,int pad_h=0,int pad_w=0,
                          int dil_h=1,int dil_w=1)
{
  auto X  = py::array_t<float, py::array::c_style | py::array::forcecast>(X_in);
  auto W  = py::array_t<float, py::array::c_style | py::array::forcecast>(W_in);
  auto dY = py::array_t<float, py::array::c_style | py::array::forcecast>(dY_in);
  if (X.ndim()!=4 || W.ndim()!=4 || dY.ndim()!=4)
    throw std::runtime_error("conv2d_backward: X,W,dY must be 4D");

  const int64_t N=X.shape(0), Cin=X.shape(1), H=X.shape(2), Ww=X.shape(3);
  const int64_t Cout=W.shape(0), WCin=W.shape(1), Kh=W.shape(2), Kw=W.shape(3);
  if (WCin!=Cin) throw std::runtime_error("conv2d_backward: Cin mismatch");
  const int64_t Ho=dY.shape(2), Wo=dY.shape(3);

  float *dX=nullptr,*dWc=nullptr,*ddY=nullptr,*dWout=nullptr,*dBout=nullptr,*dXout=nullptr;
  cudaMalloc(&dX,  sizeof(float)*N*Cin*H*Ww);
  cudaMalloc(&dWc, sizeof(float)*Cout*Cin*Kh*Kw);
  cudaMalloc(&ddY, sizeof(float)*N*Cout*Ho*Wo);
  cudaMemcpy(dX,  X.data(),  sizeof(float)*N*Cin*H*Ww,  cudaMemcpyHostToDevice);
  cudaMemcpy(dWc, W.data(),  sizeof(float)*Cout*Cin*Kh*Kw, cudaMemcpyHostToDevice);
  cudaMemcpy(ddY, dY.data(), sizeof(float)*N*Cout*Ho*Wo,   cudaMemcpyHostToDevice);

  ai::Tensor tX {dX,  make_desc_4d_nchw(N,Cin,H,Ww), ai::Device::CUDA, 0};
  ai::Tensor tW {dWc, {ai::DType::F32, ai::Layout::RowMajor, {Cout,Cin,Kh,Kw},
                       {Cin*Kh*Kw, Kh*Kw, Kw, 1}}, ai::Device::CUDA, 0};
  ai::Tensor tdY{ddY, make_desc_4d_nchw(N,Cout,Ho,Wo), ai::Device::CUDA, 0};

  ai::Tensor t_dW{}, t_dB{}, t_dX{};
  ai::Tensor *p_dW=nullptr, *p_dB=nullptr, *p_dX=nullptr;
  if (need_dW) { cudaMalloc(&dWout, sizeof(float)*Cout*Cin*Kh*Kw);
                 t_dW = {dWout, {ai::DType::F32, ai::Layout::RowMajor, {Cout,Cin,Kh,Kw},
                                   {Cin*Kh*Kw, Kh*Kw, Kw, 1}},
                         ai::Device::CUDA, 0};
                 p_dW = &t_dW; }
  if (need_dB) { cudaMalloc(&dBout, sizeof(float)*Cout);
                 t_dB = {dBout, make_desc_1d_f32(Cout), ai::Device::CUDA, 0};
                 p_dB = &t_dB; }
  if (need_dX) { cudaMalloc(&dXout, sizeof(float)*N*Cin*H*Ww);
                 t_dX = {dXout, make_desc_4d_nchw(N,Cin,H,Ww), ai::Device::CUDA, 0};
                 p_dX = &t_dX; }

  ai::Conv2DAttrs attrs{};
  attrs.stride_h=stride_h; attrs.stride_w=stride_w;
  attrs.pad_h=pad_h;       attrs.pad_w=pad_w;
  attrs.dil_h=dil_h;       attrs.dil_w=dil_w;
  attrs.groups=1;

  const int rc = ai::ops::conv2d_backward_run(tX, tW, tdY, p_dW, p_dB, p_dX, attrs, /*stream*/nullptr);
  cudaDeviceSynchronize();
  if (rc!=0) {
    if (dWout) cudaFree(dWout);
    if (dBout) cudaFree(dBout);
    if (dXout) cudaFree(dXout);
    cudaFree(dX); cudaFree(dWc); cudaFree(ddY);
    throw std::runtime_error("conv2d_backward_run failed with code " + std::to_string(rc));
  }

  py::object out_dW = py::none(), out_dB = py::none(), out_dX = py::none();
  if (dWout) { auto A = py::array_t<float>({Cout,Cin,Kh,Kw});
               cudaMemcpy(A.mutable_data(), dWout, sizeof(float)*Cout*Cin*Kh*Kw, cudaMemcpyDeviceToHost);
               out_dW = std::move(A); cudaFree(dWout); }
  if (dBout) { auto B = py::array_t<float>({Cout});
               cudaMemcpy(B.mutable_data(), dBout, sizeof(float)*Cout, cudaMemcpyDeviceToHost);
               out_dB = std::move(B); cudaFree(dBout); }
  if (dXout) { auto Xg = py::array_t<float>({N,Cin,H,Ww});
               cudaMemcpy(Xg.mutable_data(), dXout, sizeof(float)*N*Cin*H*Ww, cudaMemcpyDeviceToHost);
               out_dX = std::move(Xg); cudaFree(dXout); }

  cudaFree(dX); cudaFree(dWc); cudaFree(ddY);
  return py::make_tuple(out_dW, out_dB, out_dX);
}

// bindings/py_api.cpp  (함수 구현부에 추가: 간단 버전)
py::tuple maxpool2d(py::array X_in,
                    int kH=2,int kW=2,int sH=2,int sW=2,
                    int pH=0,int pW=0,int dH=1,int dW=1,
                    bool ceil_mode=false, bool return_indices=true)
{
  // X: (N,C,H,W) float32
  auto X = py::array_t<float, py::array::c_style | py::array::forcecast>(X_in);
  if (X.ndim()!=4) throw std::runtime_error("maxpool2d: X must be 4D (NCHW)");
  const int64_t N=X.shape(0), C=X.shape(1), H=X.shape(2), W=X.shape(3);

  // ✅ 정식 공식으로 출력 크기 계산
  int Ho_i=0, Wo_i=0;
  pool2d_output_dims_host((int)H,(int)W, kH,kW,sH,sW,pH,pW,dH,dW, ceil_mode, Ho_i, Wo_i);
  const int64_t Ho = Ho_i, Wo = Wo_i;

  // device alloc
  float *dX=nullptr,*dY=nullptr; int32_t *dInd=nullptr;
  checkCuda(cudaMalloc(&dX, sizeof(float)*N*C*H*W), "cudaMalloc X");
  checkCuda(cudaMalloc(&dY, sizeof(float)*N*C*Ho*Wo), "cudaMalloc Y");
  checkCuda(cudaMemcpy(dX, X.data(), sizeof(float)*N*C*H*W, cudaMemcpyHostToDevice), "H2D X");
  if (return_indices) checkCuda(cudaMalloc(&dInd, sizeof(int32_t)*N*C*Ho*Wo), "cudaMalloc Ind");

  // wrap
  auto mk4 = [&](int64_t n,int64_t c,int64_t h,int64_t w){
    ai::TensorDesc d{}; d.dtype=ai::DType::F32; d.layout=ai::Layout::RowMajor;
    d.shape={n,c,h,w}; d.stride={c*h*w,h*w,w,1}; return d;
  };
  auto mk4i = [&](int64_t n,int64_t c,int64_t h,int64_t w){
    ai::TensorDesc d{}; d.dtype=ai::DType::I32; d.layout=ai::Layout::RowMajor;
    d.shape={n,c,h,w}; d.stride={c*h*w,h*w,w,1}; return d;
  };
  ai::Tensor tX{dX, mk4(N,C,H,W), ai::Device::CUDA, 0};
  ai::Tensor tY{dY, mk4(N,C,Ho,Wo), ai::Device::CUDA, 0};
  ai::Tensor tInd{}; ai::Tensor* pInd=nullptr;
  if (return_indices){ tInd = ai::Tensor{dInd, mk4i(N,C,Ho,Wo), ai::Device::CUDA, 0}; pInd = &tInd; }

  ai::Pool2DAttrs a{}; a.kH=kH;a.kW=kW;a.sH=sH;a.sW=sW;a.pH=pH;a.pW=pW;a.dH=dH;a.dW=dW;a.ceil_mode=ceil_mode;

  int rc = ai::ops::maxpool2d_run(tX, tY, pInd, a, nullptr);
  checkCuda(cudaDeviceSynchronize(), "sync after maxpool2d_run");
  if (rc!=0) { if (dInd) cudaFree(dInd); cudaFree(dX); cudaFree(dY); throw std::runtime_error("maxpool2d_run failed"); }

  auto Y = py::array_t<float>({ (py::ssize_t)N,(py::ssize_t)C,(py::ssize_t)Ho,(py::ssize_t)Wo });
  checkCuda(cudaMemcpy(Y.mutable_data(), dY, sizeof(float)*N*C*Ho*Wo, cudaMemcpyDeviceToHost), "D2H Y");

  py::object Ind_py = py::none();
  if (return_indices) {
    auto Ind = py::array_t<int32_t>({ (py::ssize_t)N,(py::ssize_t)C,(py::ssize_t)Ho,(py::ssize_t)Wo });
    checkCuda(cudaMemcpy(Ind.mutable_data(), dInd, sizeof(int32_t)*N*C*Ho*Wo, cudaMemcpyDeviceToHost), "D2H Ind");
    Ind_py = std::move(Ind);
  }
  if (dInd) cudaFree(dInd); cudaFree(dX); cudaFree(dY);
  return py::make_tuple(std::move(Y), Ind_py);
}

py::array maxpool2d_backward(py::array dY_in, py::array Ind_in,
                             int H, int W,
                             int kH=2,int kW=2,int sH=2,int sW=2,
                             int pH=0,int pW=0,int dH=1,int dW=1,
                             bool ceil_mode=false)
{
  auto dY  = py::array_t<float,   py::array::c_style | py::array::forcecast>(dY_in);
  auto Ind = py::array_t<int32_t, py::array::c_style | py::array::forcecast>(Ind_in);
  if (dY.ndim()!=4 || Ind.ndim()!=4)
    throw std::runtime_error("maxpool2d_backward: dY/Ind must be 4D");

  const int64_t N  = dY.shape(0);
  const int64_t C  = dY.shape(1);
  const int64_t Ho = dY.shape(2);
  const int64_t Wo = dY.shape(3);

  float   *ddY = nullptr, *ddX = nullptr;
  int32_t *dInd = nullptr;

  checkCuda(cudaMalloc(&ddY,  sizeof(float)  * N * C * Ho * Wo), "cudaMalloc dY");
  checkCuda(cudaMalloc(&ddX,  sizeof(float)  * N * C * H  * W ), "cudaMalloc dX");
  checkCuda(cudaMalloc(&dInd, sizeof(int32_t)* N * C * Ho * Wo), "cudaMalloc Ind");

  // ✅ atomicAdd 대상인 dX는 반드시 0으로 초기화
  checkCuda(cudaMemset(ddX, 0, sizeof(float) * N * C * H * W), "memset dX=0");

  checkCuda(cudaMemcpy(ddY,  dY.data(),  sizeof(float)   * N * C * Ho * Wo, cudaMemcpyHostToDevice), "H2D dY");
  checkCuda(cudaMemcpy(dInd, Ind.data(), sizeof(int32_t) * N * C * Ho * Wo, cudaMemcpyHostToDevice), "H2D Ind");

  auto mk4 = [&](int64_t n,int64_t c,int64_t h,int64_t w){
    ai::TensorDesc d{}; d.dtype=ai::DType::F32; d.layout=ai::Layout::RowMajor;
    d.shape={n,c,h,w}; d.stride={c*h*w,h*w,w,1}; return d;
  };
  auto mk4i = [&](int64_t n,int64_t c,int64_t h,int64_t w){
    ai::TensorDesc d{}; d.dtype=ai::DType::I32; d.layout=ai::Layout::RowMajor;
    d.shape={n,c,h,w}; d.stride={c*h*w,h*w,w,1}; return d;
  };

  ai::Tensor tdY {ddY,  mk4 (N,C,Ho,Wo), ai::Device::CUDA, 0};
  ai::Tensor tInd{dInd, mk4i(N,C,Ho,Wo), ai::Device::CUDA, 0};
  ai::Tensor tdX {ddX,  mk4 (N,C,H, W ), ai::Device::CUDA, 0};

  ai::Pool2DAttrs a{};
  a.kH=kH; a.kW=kW; a.sH=sH; a.sW=sW; a.pH=pH; a.pW=pW; a.dH=dH; a.dW=dW; a.ceil_mode=ceil_mode;

  const int rc = ai::ops::maxpool2d_backward_run(tdY, tInd, tdX, a, /*stream*/nullptr);
  checkCuda(cudaDeviceSynchronize(), "sync after maxpool2d_backward_run");
  if (rc != 0) {
    cudaFree(ddY); cudaFree(dInd); cudaFree(ddX);
    throw std::runtime_error("maxpool2d_backward_run failed");
  }

  auto dX = py::array_t<float>({ (py::ssize_t)N, (py::ssize_t)C, (py::ssize_t)H, (py::ssize_t)W });
  checkCuda(cudaMemcpy(dX.mutable_data(), ddX, sizeof(float)*N*C*H*W, cudaMemcpyDeviceToHost), "D2H dX");

  cudaFree(ddY); cudaFree(dInd); cudaFree(ddX);
  return dX;
}


// === AvgPool2D: Forward ===
py::array avgpool2d(py::array X_in,
                    int kH=2,int kW=2,
                    int sH=2,int sW=2,
                    int pH=0,int pW=0,
                    int dH=1,int dW=1,
                    bool ceil_mode=false,
                    bool count_include_pad=false)
{
  auto X = py::array_t<float, py::array::c_style | py::array::forcecast>(X_in);
  if (X.ndim()!=4) throw std::runtime_error("avgpool2d: X must be 4D (NCHW)");
  const int64_t N=X.shape(0), C=X.shape(1), H=X.shape(2), W=X.shape(3);

  // ✅ 정식 공식으로 출력 크기 계산
  int Ho_i=0, Wo_i=0;
  pool2d_output_dims_host((int)H,(int)W, kH,kW,sH,sW,pH,pW,dH,dW, ceil_mode, Ho_i, Wo_i);
  const int64_t Ho = Ho_i, Wo = Wo_i;

  // device alloc/copy
  float *dX=nullptr, *dY=nullptr;
  checkCuda(cudaMalloc(&dX, sizeof(float)*N*C*H*W), "cudaMalloc X");
  checkCuda(cudaMalloc(&dY, sizeof(float)*N*C*Ho*Wo), "cudaMalloc Y");
  checkCuda(cudaMemcpy(dX, X.data(), sizeof(float)*N*C*H*W, cudaMemcpyHostToDevice), "H2D X");

  auto mk4 = [&](int64_t n,int64_t c,int64_t h,int64_t w){
    ai::TensorDesc d{}; d.dtype=ai::DType::F32; d.layout=ai::Layout::RowMajor;
    d.shape={n,c,h,w}; d.stride={c*h*w,h*w,w,1}; return d;
  };

  ai::Tensor tX{dX, mk4(N,C,H,W), ai::Device::CUDA, 0};
  ai::Tensor tY{dY, mk4(N,C,Ho,Wo), ai::Device::CUDA, 0};

  ai::Pool2DAttrs a{};
  a.kH=kH; a.kW=kW; a.sH=sH; a.sW=sW; a.pH=pH; a.pW=pW; a.dH=dH; a.dW=dW;
  a.ceil_mode=ceil_mode; a.count_include_pad=count_include_pad;

  int rc = ai::ops::avgpool2d_run(tX, tY, a, /*stream*/nullptr);
  checkCuda(cudaDeviceSynchronize(), "sync after avgpool2d_run");
  if (rc!=0) { cudaFree(dX); cudaFree(dY); throw std::runtime_error("avgpool2d_run failed"); }

  auto Y = py::array_t<float>({ (py::ssize_t)N,(py::ssize_t)C,(py::ssize_t)Ho,(py::ssize_t)Wo });
  checkCuda(cudaMemcpy(Y.mutable_data(), dY, sizeof(float)*N*C*Ho*Wo, cudaMemcpyDeviceToHost), "D2H Y");
  cudaFree(dX); cudaFree(dY);
  return Y;
}

// === AvgPool2D: Backward ===
py::array avgpool2d_backward(py::array dY_in,
                             int H, int W,
                             int kH=2,int kW=2,
                             int sH=2,int sW=2,
                             int pH=0,int pW=0,
                             int dH=1,int dW=1,
                             bool ceil_mode=false,
                             bool count_include_pad=false)
{
  auto dY = py::array_t<float, py::array::c_style | py::array::forcecast>(dY_in);
  if (dY.ndim()!=4) throw std::runtime_error("avgpool2d_backward: dY must be 4D (NCHW)");
  const int64_t N=dY.shape(0), C=dY.shape(1), Ho=dY.shape(2), Wo=dY.shape(3);

  float *ddY=nullptr, *ddX=nullptr;
  checkCuda(cudaMalloc(&ddY, sizeof(float)*N*C*Ho*Wo), "cudaMalloc dY");
  checkCuda(cudaMalloc(&ddX, sizeof(float)*N*C*H*W), "cudaMalloc dX");
  checkCuda(cudaMemcpy(ddY, dY.data(), sizeof(float)*N*C*Ho*Wo, cudaMemcpyHostToDevice), "H2D dY");

  auto mk4 = [&](int64_t n,int64_t c,int64_t h,int64_t w){
    ai::TensorDesc d{}; d.dtype=ai::DType::F32; d.layout=ai::Layout::RowMajor;
    d.shape={n,c,h,w}; d.stride={c*h*w,h*w,w,1}; return d;
  };

  ai::Tensor tdY{ddY, mk4(N,C,Ho,Wo), ai::Device::CUDA, 0};
  ai::Tensor tdX{ddX, mk4(N,C,H,W),   ai::Device::CUDA, 0};

  ai::Pool2DAttrs a{};
  a.kH=kH; a.kW=kW; a.sH=sH; a.sW=sW; a.pH=pH; a.pW=pW; a.dH=dH; a.dW=dW;
  a.ceil_mode=ceil_mode; a.count_include_pad=count_include_pad;

  int rc = ai::ops::avgpool2d_backward_run(tdY, tdX, a, /*stream*/nullptr);
  checkCuda(cudaDeviceSynchronize(), "sync after avgpool2d_backward_run");
  if (rc!=0) { cudaFree(ddY); cudaFree(ddX); throw std::runtime_error("avgpool2d_backward_run failed"); }

  auto dX = py::array_t<float>({ (py::ssize_t)N, (py::ssize_t)C, (py::ssize_t)H, (py::ssize_t)W });
  checkCuda(cudaMemcpy(dX.mutable_data(), ddX, sizeof(float)*N*C*H*W, cudaMemcpyDeviceToHost), "D2H dX");
  cudaFree(ddY); cudaFree(ddX);
  return dX;
}

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <cctype>

#include "ai/tensor.hpp"
#include "backends/cuda/ops/elementwise/api.hpp"

namespace py = pybind11;

// ---- Elementwise: Unary ----
py::array ewise_unary(py::array X_in, std::string kind = "relu",
                      double alpha = 0.01, double clip_min = -1e30, double clip_max = 1e30)
{
  auto X = py::array_t<float, py::array::c_style | py::array::forcecast>(X_in);
  if (X.ndim() < 1) throw std::runtime_error("ewise_unary: X must have >=1 dim");

  int64_t nElem = 1;
  std::vector<int64_t> shape_i64(X.ndim());
  for (py::ssize_t i=0; i<X.ndim(); ++i) {
      shape_i64[i] = static_cast<int64_t>(X.shape()[i]);
      nElem *= shape_i64[i];
  }

  float *dX=nullptr, *dY=nullptr;
  cudaMalloc(&dX, sizeof(float)*nElem);
  cudaMalloc(&dY, sizeof(float)*nElem);
  cudaMemcpy(dX, X.data(), sizeof(float)*nElem, cudaMemcpyHostToDevice);

  ai::TensorDesc desc{};
  desc.dtype = ai::DType::F32;
  desc.layout = ai::Layout::RowMajor;
  desc.shape = shape_i64;
  desc.stride.resize(shape_i64.size());
  {
    int64_t s = 1;
    for (int i=(int)shape_i64.size()-1; i>=0; --i) {
      desc.stride[i] = s;
      s *= shape_i64[i];
    }
  }

  ai::Tensor tX{dX, desc, ai::Device::CUDA, 0};
  ai::Tensor tY{dY, desc, ai::Device::CUDA, 0};

  auto to_unary = [](const std::string& s) -> ai::UnaryOp {
    std::string k; k.reserve(s.size());
    for (char c: s) k.push_back((char)std::tolower((unsigned char)c));
    if (k=="identity") return ai::UnaryOp::Identity;
    if (k=="relu")     return ai::UnaryOp::ReLU;
    if (k=="leakyrelu"||k=="leaky_relu"||k=="lrelu") return ai::UnaryOp::LeakyReLU;
    if (k=="sigmoid")  return ai::UnaryOp::Sigmoid;
    if (k=="tanh")     return ai::UnaryOp::Tanh;
    if (k=="gelu")     return ai::UnaryOp::GELU;
    if (k=="exp")      return ai::UnaryOp::Exp;
    if (k=="log")      return ai::UnaryOp::Log;
    if (k=="sqrt")     return ai::UnaryOp::Sqrt;
    if (k=="rsqrt")    return ai::UnaryOp::Rsqrt;
    if (k=="clip")     return ai::UnaryOp::Clip;
    throw std::runtime_error("ewise_unary: unknown kind: " + s);
  };

  ai::EWiseUnaryAttrs attrs{};
  attrs.alpha    = (float)alpha;
  attrs.clip_min = (float)clip_min;
  attrs.clip_max = (float)clip_max;

  auto st = ai::EWiseUnaryCudaLaunch(tX, tY, to_unary(kind), attrs, nullptr);
  cudaError_t sync_st = cudaDeviceSynchronize();
  if (st != ai::Status::Ok || sync_st != cudaSuccess) {
    cudaFree(dX); cudaFree(dY);
    throw std::runtime_error("EWiseUnaryCudaLaunch failed");
  }

  // 🔧 pybind11 array 생성 (size_t 기반 shape)
  std::vector<size_t> shape;
  shape.reserve(shape_i64.size());
  for (auto v : shape_i64) shape.push_back(static_cast<size_t>(v));

  auto Y = py::array_t<float>(shape);
  cudaMemcpy(Y.mutable_data(), dY, sizeof(float)*nElem, cudaMemcpyDeviceToHost);
  cudaFree(dX); cudaFree(dY);
  return Y;
}

// ---- Elementwise: Binary ----
py::array ewise_binary(py::array A_in, py::array B_in, std::string kind = "add",
                       double alpha = 1.0, double beta = 1.0)
{
  auto A = py::array_t<float, py::array::c_style | py::array::forcecast>(A_in);
  auto B = py::array_t<float, py::array::c_style | py::array::forcecast>(B_in);
  if (A.ndim()<1 || B.ndim()<1) throw std::runtime_error("ewise_binary: A,B must have >=1 dim");

  const py::ssize_t nd = std::max(A.ndim(), B.ndim());
  std::vector<int64_t> out(nd, 1);
  for (py::ssize_t i=0; i<nd; ++i) {
    int64_t a = (i < nd - A.ndim()) ? 1 : (int64_t)A.shape(i - (nd - A.ndim()));
    int64_t b = (i < nd - B.ndim()) ? 1 : (int64_t)B.shape(i - (nd - B.ndim()));
    if (a!=b && a!=1 && b!=1) throw std::runtime_error("ewise_binary: shapes not broadcastable");
    out[i] = (a==1)? b : a;
  }
  int64_t nElem = 1; for (auto v: out) nElem *= v;

  float *dA=nullptr, *dB=nullptr, *dY=nullptr;
  cudaMalloc(&dA, sizeof(float)*(int64_t)A.size());
  cudaMalloc(&dB, sizeof(float)*(int64_t)B.size());
  cudaMalloc(&dY, sizeof(float)*nElem);
  cudaMemcpy(dA, A.data(), sizeof(float)*(int64_t)A.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B.data(), sizeof(float)*(int64_t)B.size(), cudaMemcpyHostToDevice);

  auto mk_desc = [](const std::vector<int64_t>& s){
    ai::TensorDesc d{}; d.dtype=ai::DType::F32; d.layout=ai::Layout::RowMajor;
    d.shape = s; d.stride.resize(s.size());
    int64_t st=1; for (int i=(int)s.size()-1;i>=0;--i){ d.stride[i]=st; st*=s[i]; }
    return d;
  };
  auto vec_i64 = [](const py::array& arr){
    std::vector<int64_t> v(arr.ndim());
    for (py::ssize_t i=0;i<arr.ndim();++i) v[i] = (int64_t)arr.shape(i);
    return v;
  };

  ai::Tensor tA{dA, mk_desc(vec_i64(A)), ai::Device::CUDA, 0};
  ai::Tensor tB{dB, mk_desc(vec_i64(B)), ai::Device::CUDA, 0};
  ai::Tensor tY{dY, mk_desc(out),        ai::Device::CUDA, 0};

  auto to_binary = [](const std::string& s) -> ai::BinaryOp {
    std::string k; k.reserve(s.size());
    for (char c: s) k.push_back((char)std::tolower((unsigned char)c));
    if (k=="add") return ai::BinaryOp::Add;
    if (k=="sub") return ai::BinaryOp::Sub;
    if (k=="mul") return ai::BinaryOp::Mul;
    if (k=="div") return ai::BinaryOp::Div;
    if (k=="max") return ai::BinaryOp::Max;
    if (k=="min") return ai::BinaryOp::Min;
    if (k=="pow") return ai::BinaryOp::Pow;
    throw std::runtime_error("ewise_binary: unknown kind: " + s);
  };

  ai::EWiseBinaryAttrs attrs{};
  attrs.alpha = (float)alpha;
  attrs.beta  = (float)beta;

  auto st = ai::EWiseBinaryCudaLaunch(tA, tB, tY, to_binary(kind), attrs, nullptr);
  cudaError_t sync_st = cudaDeviceSynchronize();
  if (st != ai::Status::Ok || sync_st != cudaSuccess){
    cudaFree(dA); cudaFree(dB); cudaFree(dY);
    throw std::runtime_error("EWiseBinaryCudaLaunch failed");
  }

  // 🔧 pybind11 array 생성 (size_t 기반 shape)
  std::vector<size_t> shape;
  shape.reserve(out.size());
  for (auto v: out) shape.push_back(static_cast<size_t>(v));

  auto Y = py::array_t<float>(shape);
  cudaMemcpy(Y.mutable_data(), dY, sizeof(float)*nElem, cudaMemcpyDeviceToHost);
  cudaFree(dA); cudaFree(dB); cudaFree(dY);
  return Y;
}


// 공통 reduce 실행기
static py::array do_reduce(py::array X_in,
                           py::object axes_in,
                           bool keepdim,
                           ai::ReduceOp op)
{
  auto X = py::array_t<float, py::array::c_style | py::array::forcecast>(X_in);
  const int nd = X.ndim();

  // axes 파싱
  std::vector<int> axes;
  if (!axes_in.is_none()) {
    for (auto v: axes_in.cast<std::vector<int>>())
      axes.push_back(v);
  } else {
    axes.resize(nd);
    for (int i=0;i<nd;i++) axes[i]=i;
  }

  // 입력 shape
  int64_t elems=1;
  std::vector<int64_t> xshape(nd);
  for (int i=0;i<nd;i++){ xshape[i]=X.shape(i); elems*=xshape[i]; }

  // Host→Device
  float *dX=nullptr, *dY=nullptr;
  checkCuda(cudaMalloc(&dX, sizeof(float)*elems), "malloc dX");
  checkCuda(cudaMemcpy(dX, X.data(), sizeof(float)*elems, cudaMemcpyHostToDevice), "H2D X");

  // 출력 shape
  auto yshape = reduced_shape(xshape, axes, keepdim);
  int64_t y_elems=1; for (auto v: yshape) y_elems*=v;
  checkCuda(cudaMalloc(&dY, sizeof(float)*y_elems), "malloc dY");

  // Tensor wrap
  ai::TensorDesc dx{}; dx.dtype=ai::DType::F32; dx.layout=ai::Layout::RowMajor; dx.shape=xshape;
  ai::Tensor tX{dX, dx, ai::Device::CUDA, 0};
  ai::TensorDesc dy{}; dy.dtype=ai::DType::F32; dy.layout=ai::Layout::RowMajor; dy.shape=yshape;
  ai::Tensor tY{dY, dy, ai::Device::CUDA, 0};

  // attrs
  ai::ReduceAttrs a{};
  a.axes = axes;
  a.keepdim = keepdim;
  a.op = op;

  auto st = ai::ReduceCudaLaunch(tX, tY, a, nullptr);
  checkCuda(cudaDeviceSynchronize(), "sync reduce");
  if (st != ai::Status::Ok){
    cudaFree(dX); cudaFree(dY);
    throw std::runtime_error("ReduceCudaLaunch failed");
  }

  // 결과 복사
  std::vector<size_t> yshape_size_t(yshape.begin(), yshape.end());
  auto Y = py::array_t<float>(yshape_size_t);
  checkCuda(cudaMemcpy(Y.mutable_data(), dY, sizeof(float)*y_elems, cudaMemcpyDeviceToHost), "D2H Y");

  cudaFree(dX); cudaFree(dY);
  return Y;
}

// ----------------- API wrappers -----------------

py::array reduce_sum(py::array X,
                     std::optional<std::vector<int>> axes = std::nullopt,
                     bool keepdim=false)
{
    return do_reduce(X, axes.has_value() ? py::cast(axes.value()) : py::none(),
                     keepdim, ai::ReduceOp::Sum);
}

py::array reduce_mean(py::array X,
                      std::optional<std::vector<int>> axes = std::nullopt,
                      bool keepdim=false)
{
    return do_reduce(X, axes.has_value() ? py::cast(axes.value()) : py::none(),
                     keepdim, ai::ReduceOp::Mean);
}

py::array reduce_max(py::array X,
                     std::optional<std::vector<int>> axes = std::nullopt,
                     bool keepdim=false)
{
    return do_reduce(X, axes.has_value() ? py::cast(axes.value()) : py::none(),
                     keepdim, ai::ReduceOp::Max);
}

py::array reduce_min(py::array X,
                     std::optional<std::vector<int>> axes = std::nullopt,
                     bool keepdim=false)
{
    return do_reduce(X, axes.has_value() ? py::cast(axes.value()) : py::none(),
                     keepdim, ai::ReduceOp::Min);
}


// -------------------- Module --------------------
PYBIND11_MODULE(_core, m) {
  // 커널 등록 보장
  ai_backend_cuda_register_all();

  m.doc() = "graph_executor_v2 python bindings";

  // FWD 단발
  m.def("gemm_bias_act", &gemm_bias_act,
        py::arg("A"), py::arg("B"), py::arg("bias") = py::none(),
        py::arg("act") = "relu", py::arg("leaky_slope") = 0.01,
R"(GEMM + optional bias + activation (CUDA, f32, row-major)
A: (M,K) float32 contiguous
B: (K,N) float32 contiguous
bias: 1D (1|M|N) float32 or None
act: one of ["none","relu","leaky_relu","gelu","sigmoid","tanh"])");

  // FWD with Z
  m.def("gemm_bias_act_fwd_with_z", &gemm_bias_act_fwd_with_z,
        py::arg("A"), py::arg("B"), py::arg("bias")=py::none(),
        py::arg("act")="relu", py::arg("leaky_slope")=0.01,
        "Forward with Z stash: returns (Y, Z).");

  // BWD
  m.def("gemm_bias_act_bwd", &gemm_bias_act_bwd,
        py::arg("A"), py::arg("B"), py::arg("gY"), py::arg("Z"),
        py::arg("act")="relu", py::arg("bias_kind")="none", py::arg("leaky_slope")=0.01,
        "Backward: returns dict with gA, gB, gBias, gC(None).");

  m.def("rmsnorm", &rmsnorm,
        py::arg("X"), py::arg("gamma")=py::none(), py::arg("beta")=py::none(),
        py::arg("eps")=1e-6, "RMSNorm (CUDA, f32, row-major)");

  m.def("rmsnorm_backward", &rmsnorm_backward,
        py::arg("X"), py::arg("gamma")=py::none(), py::arg("dY"),
        py::arg("eps")=1e-6, "RMSNorm backward (CUDA)");
  
  m.def("layernorm", &layernorm,
        py::arg("X"), py::arg("gamma")=py::none(), py::arg("beta")=py::none(),
        py::arg("eps")=1e-5, "LayerNorm (CUDA, f32, row-major)");

  m.def("layernorm_backward", &layernorm_backward,
        py::arg("X"), py::arg("gamma")=py::none(), py::arg("dY"),
        py::arg("eps")=1e-5, "LayerNorm backward (CUDA)");
  
  // --- Softmax (행 단위) ---
  m.def("softmax", &softmax,
        py::arg("X"),
        py::arg("mask") = py::none(),   // [M,N] 또는 [N], 값은 x에 더해짐 (e.g., -inf 또는 0)
        py::arg("scale") = 1.0,         // y = softmax(scale * (x + mask))
        py::arg("log") = false,         // true면 logsoftmax 결과 반환
  R"(Row-wise Softmax / LogSoftmax (CUDA, f32, row-major)
  X: 2D ndarray (M,N), float32, contiguous
  mask: optional [M,N] or [N] (broadcast) float32; values are added to X before softmax
  scale: multiply inputs before softmax (e.g., 1/T)
  log: if true, returns logsoftmax
  Returns: Y with shape (M,N))");

  m.def("softmax_backward", &softmax_backward,
        py::arg("Y"),                  // softmax(X) 결과(또는 log=false 기준의 확률)
        py::arg("dY"),                 // loss wrt Y
        py::arg("log") = false,        // forward에 log=True를 썼다면 동일하게 True로
  R"(Backward of (Log)Softmax (CUDA)
  Inputs:
    Y : forward softmax outputs (shape [M,N], not log-prob; if you used log=True forward, pass softmax probs here)
    dY: gradient w.r.t. Y
    log: set to True if forward used logsoftmax (formula changes)
  Returns:
    dX with shape (M,N))");

  // --- 편의 함수(선택): logsoftmax 별도 이름으로 노출하고 싶다면 ---
  m.def("logsoftmax",
        [](py::array X, py::object mask = py::none(), double scale = 1.0){
          return softmax(X, mask, scale, /*log=*/true);
        },
        py::arg("X"), py::arg("mask") = py::none(), py::arg("scale") = 1.0,
  R"(Convenience wrapper for log-softmax(X, mask, scale))");

  m.def("logsoftmax_backward",
        [](py::array Y_softmax, py::array dY){   // 주의: logsoftmax의 bwd는 softmax 확률이 필요
          return softmax_backward(Y_softmax, dY, /*log=*/true);
        },
        py::arg("Y_softmax"), py::arg("dY"),
  R"(Backward wrapper for log-softmax.
  NOTE: Pass softmax probabilities (not log-probs) as Y_softmax.)");


  // 등록부:
  m.def("cross_entropy", &cross_entropy,
        py::arg("X"), py::arg("target"),
        py::arg("reduction")="mean",
        py::arg("ignore_index")=-1,
        py::arg("label_smoothing")=0.0,
        "CrossEntropy with logits (CUDA, options: reduction/ignore_index/label_smoothing)");

  m.def("cross_entropy_backward", &cross_entropy_backward,
        py::arg("X"), py::arg("target"),
        py::arg("reduction")="mean",
        py::arg("ignore_index")=-1,
        py::arg("label_smoothing")=0.0,
        "Backward of CrossEntropy (same options)");

  m.def("dropout", &dropout,
        py::arg("X"), py::arg("p")=0.1, py::arg("return_mask")=false,
        py::arg("seed")=0x1234, py::arg("scale_in_train")=true,
        "Dropout forward (CUDA, f32). Returns Y or (Y, mask).");

  m.def("dropout_backward", &dropout_backward,
        py::arg("dY"), py::arg("mask"), py::arg("p")=0.1,
        py::arg("seed")=0x1234, py::arg("scale_in_train")=true,
        "Dropout backward (CUDA): dX = dY * mask * scale.");


  m.def("conv2d", &conv2d,
        py::arg("X"), py::arg("W"), py::arg("B") = py::none(),
        py::arg("stride_h")=1, py::arg("stride_w")=1,
        py::arg("pad_h")=0,    py::arg("pad_w")=0,
        py::arg("dil_h")=1,    py::arg("dil_w")=1,
  R"(Conv2D forward (CUDA, NCHW, F32, groups=1)
  X: (N,Cin,H,W), W: (Cout,Cin,Kh,Kw), B: (Cout) or None
  Returns Y: (N,Cout,Ho,Wo))");

  m.def("conv2d_backward", &conv2d_backward,
        py::arg("X"), py::arg("W"), py::arg("dY"),
        py::arg("need_dW")=true, py::arg("need_dB")=true, py::arg("need_dX")=true,
        py::arg("stride_h")=1, py::arg("stride_w")=1,
        py::arg("pad_h")=0,    py::arg("pad_w")=0,
        py::arg("dil_h")=1,    py::arg("dil_w")=1,
  R"(Conv2D backward (CUDA, NCHW, F32, groups=1)
  Returns tuple (dW, dB, dX); each item can be None depending on need_* flags.)");

  m.def("maxpool2d", &maxpool2d,
        py::arg("X"),
        py::arg("kH")=2, py::arg("kW")=2,
        py::arg("sH")=2, py::arg("sW")=2,
        py::arg("pH")=0, py::arg("pW")=0,
        py::arg("dH")=1, py::arg("dW")=1,
        py::arg("ceil_mode")=false, py::arg("return_indices")=true,
        "MaxPool2D forward (NCHW). Returns (Y, Indices or None)");

  m.def("maxpool2d_backward", &maxpool2d_backward,
        py::arg("dY"), py::arg("Indices"),
        py::arg("H"), py::arg("W"),
        py::arg("kH")=2, py::arg("kW")=2,
        py::arg("sH")=2, py::arg("sW")=2,
        py::arg("pH")=0, py::arg("pW")=0,
        py::arg("dH")=1, py::arg("dW")=1,
        py::arg("ceil_mode")=false,
        "MaxPool2D backward (NCHW).");

  m.def("avgpool2d", &avgpool2d,
        py::arg("X"),
        py::arg("kH")=2, py::arg("kW")=2,
        py::arg("sH")=2, py::arg("sW")=2,
        py::arg("pH")=0, py::arg("pW")=0,
        py::arg("dH")=1, py::arg("dW")=1,
        py::arg("ceil_mode")=false,
        py::arg("count_include_pad")=false,
        "AvgPool2D forward (NCHW).");

  m.def("avgpool2d_backward", &avgpool2d_backward,
        py::arg("dY"),
        py::arg("H"), py::arg("W"),
        py::arg("kH")=2, py::arg("kW")=2,
        py::arg("sH")=2, py::arg("sW")=2,
        py::arg("pH")=0, py::arg("pW")=0,
        py::arg("dH")=1, py::arg("dW")=1,
        py::arg("ceil_mode")=false,
        py::arg("count_include_pad")=false,
        "AvgPool2D backward (NCHW).");

  m.def("ewise_unary",  &ewise_unary,
        py::arg("X"), py::arg("kind")="relu",
        py::arg("alpha")=0.01, py::arg("clip_min")=-1e30, py::arg("clip_max")=1e30,
        "Elementwise unary (CUDA, f32, broadcasting).");

  m.def("ewise_binary", &ewise_binary,
        py::arg("A"), py::arg("B"), py::arg("kind")="add",
        py::arg("alpha")=1.0, py::arg("beta")=1.0,
        "Elementwise binary (CUDA, f32, broadcasting).");

  m.def("reduce_sum",  &reduce_sum,
        py::arg("X"), py::arg("axes")=py::none(), py::arg("keepdim")=false,
        "Sum reduction over specified axes");

  m.def("reduce_mean", &reduce_mean,
        py::arg("X"), py::arg("axes")=py::none(), py::arg("keepdim")=false,
        "Mean reduction over specified axes");

  m.def("reduce_max",  &reduce_max,
        py::arg("X"), py::arg("axes")=py::none(), py::arg("keepdim")=false,
        "Max reduction over specified axes");

  m.def("reduce_min",  &reduce_min,
        py::arg("X"), py::arg("axes")=py::none(), py::arg("keepdim")=false,
        "Min reduction over specified axes");


  m.def("sdpa",
    [](py::array q_in, py::array k_in, py::array v_in,
      py::object mask_in, double scale, bool causal,
      double dropout_p, bool scale_in_train, uint64_t seed)
    {
      // 1) NumPy → host float32 C-연속 4D 보장
      auto Qh = py::array_t<float, py::array::c_style | py::array::forcecast>(q_in);
      auto Kh = py::array_t<float, py::array::c_style | py::array::forcecast>(k_in);
      auto Vh = py::array_t<float, py::array::c_style | py::array::forcecast>(v_in);
      if (Qh.ndim()!=4 || Kh.ndim()!=4 || Vh.ndim()!=4)
        throw std::runtime_error("sdpa: q,k,v must be 4D (B,H,M/D or N/D)");

      const int64_t B  = Qh.shape(0);
      const int64_t H  = Qh.shape(1);
      const int64_t M  = Qh.shape(2);
      const int64_t D  = Qh.shape(3);
      const int64_t NB = Kh.shape(2); // N

      if (Kh.shape(0)!=B || Kh.shape(1)!=H || Kh.shape(3)!=D)
        throw std::runtime_error("sdpa: k must be [B,H,N,D] and match q on B,H,D");
      if (Vh.shape(0)!=B || Vh.shape(1)!=H || Vh.shape(2)!=NB || Vh.shape(3)!=D)
        throw std::runtime_error("sdpa: v must be [B,H,N,D] and match k on B,H,N,D");

      // 2) Device alloc & H2D
      float *dQ=nullptr, *dK=nullptr, *dV=nullptr, *dY=nullptr;
      size_t bytesQ = (size_t)B*H*M*D*sizeof(float);
      size_t bytesK = (size_t)B*H*NB*D*sizeof(float);
      size_t bytesV = (size_t)B*H*NB*D*sizeof(float);
      size_t bytesY = (size_t)B*H*M*D*sizeof(float);

      checkCuda(cudaMalloc(&dQ, bytesQ), "cudaMalloc dQ");
      checkCuda(cudaMalloc(&dK, bytesK), "cudaMalloc dK");
      checkCuda(cudaMalloc(&dV, bytesV), "cudaMalloc dV");
      checkCuda(cudaMalloc(&dY, bytesY), "cudaMalloc dY");

      checkCuda(cudaMemcpy(dQ, Qh.data(), bytesQ, cudaMemcpyHostToDevice), "H2D Q");
      checkCuda(cudaMemcpy(dK, Kh.data(), bytesK, cudaMemcpyHostToDevice), "H2D K");
      checkCuda(cudaMemcpy(dV, Vh.data(), bytesV, cudaMemcpyHostToDevice), "H2D V");

      // 3) Wrap tensors (RowMajor BHMD)
      auto make_bhxd = [](int64_t B, int64_t H, int64_t X, int64_t D){
        ai::TensorDesc d{}; d.dtype=ai::DType::F32; d.layout=ai::Layout::RowMajor;
        d.shape  = {B,H,X,D};
        d.stride = {H*X*D, X*D, D, 1};
        return d;
      };
      ai::Tensor tQ{dQ, make_bhxd(B,H,M,D), ai::Device::CUDA, 0};
      ai::Tensor tK{dK, make_bhxd(B,H,NB,D), ai::Device::CUDA, 0};
      ai::Tensor tV{dV, make_bhxd(B,H,NB,D), ai::Device::CUDA, 0};
      ai::Tensor tY{dY, make_bhxd(B,H,M,D),  ai::Device::CUDA, 0};

      // 4) mask: 아직 미구현 → None만 허용
      const ai::Tensor* pMask = nullptr;
      if (!mask_in.is_none()) {
        // 추후 [B,1,M,N]/[B,H,M,N] 브로드캐스트 지원 예정
        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dY);
        throw std::runtime_error("sdpa: mask is not implemented yet (pass None).");
      }

      // 5) attrs 채우기
      ai::SDPAAttrs a{};
      a.scale          = static_cast<float>(scale);      // 0 → 내부에서 1/sqrt(D)
      a.causal         = causal;
      a.dropout_p      = static_cast<float>(dropout_p);
      a.scale_in_train = scale_in_train;
      a.seed           = seed;

      // 6) 디스패치 호출 (동기화/에러 처리 동일 스타일)
      int rc = ai::ops::sdpa_run(tQ, tK, tV, pMask, tY, a, /*stream*/nullptr);
      checkCuda(cudaDeviceSynchronize(), "sync after sdpa_run");
      if (rc != 0) {
        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dY);
        throw std::runtime_error(std::string("sdpa_run failed with code ") + std::to_string(rc));
      }

      // 7) D2H & free
      auto Y = py::array_t<float>({B,H,M,D});
      checkCuda(cudaMemcpy(Y.mutable_data(), dY, bytesY, cudaMemcpyDeviceToHost), "D2H Y");
      cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dY);
      return Y;
    },
    py::arg("q"), py::arg("k"), py::arg("v"),
    py::arg("mask") = py::none(),
    py::arg("scale") = 0.0,        // 0이면 내부에서 1/sqrt(D)
    py::arg("causal") = false,
    py::arg("dropout_p") = 0.0,
    py::arg("scale_in_train") = true,
    py::arg("seed") = 0,
    "Scaled Dot-Product Attention (FWD)"
  );



  // GemmPlan
  py::class_<GemmPlan>(m, "GemmPlan")
    .def(py::init<int64_t,int64_t,int64_t,const std::string&,const std::string&,double>(),
         py::arg("M"), py::arg("K"), py::arg("N"),
         py::arg("act")="relu", py::arg("bias_kind")="pern",
         py::arg("leaky_slope")=0.01,
R"(Persistent GEMM plan (device buffers reused; kernel-time measured via CUDA events).
bias_kind: "none" | "pern" | "perm" | "scalar")")
    .def("upload", &GemmPlan::upload, py::arg("A"), py::arg("B"), py::arg("bias")=py::none(),
         R"(Upload host arrays to device buffers. Shapes must match plan dims.)")
    .def("run", &GemmPlan::run, py::arg("copy_out")=false, py::arg("out_array")=py::none(),
         R"(Launch once and return kernel time (ms). If copy_out=True, also perform D2H of Y (time not included).)")
    .def("get_output", &GemmPlan::get_output,
         R"(Return the last output Y as a numpy array.)")
    .def_property_readonly("M", &GemmPlan::M)
    .def_property_readonly("K", &GemmPlan::K)
    .def_property_readonly("N", &GemmPlan::N)
    .def_property_readonly("bias_len", &GemmPlan::bias_len);


}
