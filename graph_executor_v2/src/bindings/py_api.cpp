// bindings/py_api.cpp
#include <stdexcept>
#include <string>
#include <vector>
#include <cctype>

#include <cuda_runtime.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "ai/tensor.hpp"
#include "ai/dispatch.hpp"
#include "ai/op_schema.hpp"
#include "regemm/api.h"  // EX forward(Z stash) 직접 호출용
#include "backends/cuda/ops/rmsnorm/api.hpp"  // ✅ RMSNormAttrs, 런처 시그니처 출처 통일

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

static inline TensorDesc make_desc_2d(int64_t rows, int64_t cols){
  TensorDesc d{};
  d.dtype  = DType::F32;
  d.layout = Layout::RowMajor;
  d.shape  = {rows, cols};
  d.stride = {cols, 1}; // contiguous row-major
  return d;
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

// -------------------- 디스패치 엔트리(정의는 src/dispatch/registry.cpp) --------------------
namespace ai { namespace ops {
int gemm_run(const Tensor& A, const Tensor& B, const Tensor* Bias,
             Tensor& Y, const GemmAttrs& attrs, StreamHandle stream);

int gemm_bwd_run(const Tensor& A, const Tensor& B, const Tensor* C,
                 const Tensor& gY, const Tensor& Z,
                 Tensor* gA, Tensor* gB, Tensor* gC, Tensor* gBias,
                 const GemmAttrs& attrs, StreamHandle stream);

// ⬇️ RMSNorm은 attrs 타입이 ai::RMSNormAttrs 임에 유의
int rmsnorm_run(const Tensor&, const Tensor*, const Tensor*, Tensor&, const ai::RMSNormAttrs&, StreamHandle);
int rmsnorm_backward_run(const Tensor&, const Tensor*, const Tensor&, Tensor&, Tensor*, Tensor*, const ai::RMSNormAttrs&, StreamHandle);

}} // namespace ai::ops

// -------------------- FWD: 단발 함수 --------------------
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
