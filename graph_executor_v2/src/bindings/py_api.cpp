// bindings/py_api.cpp
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "ai/tensor.hpp"
#include "ai/dispatch.hpp"
#include "ai/op_schema.hpp"

extern "C" void ai_backend_cuda_register_all();


namespace py = pybind11;
using namespace ai;

// gemm 엔트리 (src/ops/gemm.cpp 에 구현)
namespace ai { namespace ops {
int gemm_run(const Tensor& A, const Tensor& B, const Tensor* Bias,
             Tensor& Y, const GemmAttrs& attrs, StreamHandle stream);
}} // namespace ai::ops

// ---- CUDA helpers ----
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

// ---- Binding: gemm_bias_act ----
py::array gemm_bias_act(py::array A_in, py::array B_in,
                        py::object bias_in = py::none(),
                        std::string act = "relu",
                        double leaky_slope = 0.01)
{
  // 1) NumPy → host float32, contiguous, 2D 체크
  auto A_f = py::array_t<float, py::array::c_style | py::array::forcecast>(A_in);
  auto B_f = py::array_t<float, py::array::c_style | py::array::forcecast>(B_in);
  if (A_f.ndim()!=2 || B_f.ndim()!=2) throw std::runtime_error("A, B must be 2D");

  const int64_t M = A_f.shape(0);
  const int64_t K = A_f.shape(1);
  const int64_t Kb= B_f.shape(0);
  const int64_t N = B_f.shape(1);
  if (K != Kb) throw std::runtime_error("shape mismatch: A[M,K] @ B[K,N]");

  // 2) Bias(optional)
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

  // 3) Device alloc/copy
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

  // 4) ai::Tensor 래핑
  Tensor tA{dA, make_desc_2d(M,K), Device::CUDA, 0};
  Tensor tB{dB, make_desc_2d(K,N), Device::CUDA, 0};
  Tensor tY{dY, make_desc_2d(M,N), Device::CUDA, 0};

  Tensor tBias{};
  Tensor* pBias = nullptr;
  if (with_bias) {
    // 1D desc: [N] or [M] or [1]
    TensorDesc bd{};
    bd.dtype = DType::F32; bd.layout = Layout::RowMajor;
    if (bias_len==N)       { bd.shape = {N}; }
    else if (bias_len==M)  { bd.shape = {M}; }
    else                   { bd.shape = {1}; }
    bd.stride = {1};
    tBias = Tensor{dBias, bd, Device::CUDA, 0};
    pBias = &tBias;
  }

  // 5) attrs 구성
  GemmAttrs attrs{};
  attrs.act = parse_act(act);
  attrs.with_bias = with_bias;
  attrs.leaky_slope = static_cast<float>(leaky_slope);

  // 6) 호출 (default stream = nullptr)
  const int rc = ai::ops::gemm_run(tA, tB, pBias, tY, attrs, /*stream*/nullptr);
  checkCuda(cudaDeviceSynchronize(), "cuda sync after gemm");
  if (rc != 0) {
    // 메모리 해제 전에 예외로 나가면 안되므로 먼저 free
    cudaFree(dA); cudaFree(dB); cudaFree(dY);
    if (dBias) cudaFree(dBias);
    throw std::runtime_error("gemm_run failed with code " + std::to_string(rc));
  }

  // 7) 결과를 NumPy로 복사
  auto Y_out = py::array_t<float>({M, N});
  checkCuda(cudaMemcpy(Y_out.mutable_data(), dY, sizeof(float)*M*N, cudaMemcpyDeviceToHost), "D2H Y");

  // 8) free
  cudaFree(dA); cudaFree(dB); cudaFree(dY);
  if (dBias) cudaFree(dBias);

  return Y_out;
}

// ---- Module ----
PYBIND11_MODULE(_core, m) {
  ai_backend_cuda_register_all();

  m.doc() = "graph_executor_v2 python bindings";
  
  m.def("gemm_bias_act", &gemm_bias_act,
        py::arg("A"), py::arg("B"), py::arg("bias") = py::none(),
        py::arg("act") = "relu", py::arg("leaky_slope") = 0.01,
        R"(GEMM + optional bias + activation (CUDA, f32, row-major)
A: (M,K) float32 contiguous
B: (K,N) float32 contiguous
bias: 1D (1|M|N) float32 or None
act: one of ["none","relu","leaky_relu","gelu","sigmoid","tanh"])");
}
