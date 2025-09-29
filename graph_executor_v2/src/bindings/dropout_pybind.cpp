// src/bindings/dropout_pybind.cpp
#include <cuda_runtime.h>
#include <stdexcept>
#include <memory>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#include "backends/cuda/ops/dropout/api.hpp"   // DropoutAttrs & ops::dropout_run decl

#include "src/ops/dropout.cpp"


namespace py = pybind11;
using namespace ai;

// ----- small helpers -----
static inline TensorDesc make_desc_2d_rowmajor(int64_t M, int64_t N, DType dt){
  TensorDesc d;
  d.dtype   = dt;
  d.layout  = Layout::RowMajor;
  d.shape   = {M, N};
  d.stride  = {N, 1};      // 요소 단위
  return d;
}

template <typename T>
struct CudaDeleter {
  void operator()(T* p) const noexcept { if (p) cudaFree(p); }
};

template <typename T>
using cuda_unique_ptr = std::unique_ptr<T, CudaDeleter<T>>;

// ----- bindings -----

// forward(X, p, return_mask, seed, scale_in_train) -> Y or (Y, mask)
static py::object dropout(py::array X_in,
                          double p = 0.1,
                          py::object return_mask = py::bool_(false),
                          std::uint64_t seed = 0x1234,
                          bool scale_in_train = true)
{
  // Force float32, C-order
  auto X = py::array_t<float, py::array::c_style | py::array::forcecast>(X_in);
  if (X.ndim() != 2) throw std::runtime_error("dropout: X must be 2D");
  const int64_t M = X.shape(0), N = X.shape(1);
  const size_t  count = static_cast<size_t>(M) * static_cast<size_t>(N);

  // device alloc
  float* dX_raw = nullptr;
  float* dY_raw = nullptr;
  cudaMalloc(&dX_raw, sizeof(float) * count);
  cudaMalloc(&dY_raw, sizeof(float) * count);
  cuda_unique_ptr<float> dX(dX_raw), dY(dY_raw);

  // H2D
  cudaMemcpy(dX.get(), X.data(), sizeof(float) * count, cudaMemcpyHostToDevice);

  // optional mask
  int32_t* dM_raw = nullptr;
  cuda_unique_ptr<int32_t> dM;
  if (py::cast<bool>(return_mask)) {
    cudaMalloc(&dM_raw, sizeof(int32_t) * count);
    dM.reset(dM_raw);
  }

  // wrap tensors
  Tensor tX{dX.get(), make_desc_2d_rowmajor(M, N, DType::F32), Device::CUDA, 0};
  Tensor tY{dY.get(), make_desc_2d_rowmajor(M, N, DType::F32), Device::CUDA, 0};
  Tensor tM{}; Tensor* pM = nullptr;
  if (dM) { tM = Tensor{dM.get(), make_desc_2d_rowmajor(M, N, DType::I32), Device::CUDA, 0}; pM = &tM; }

  DropoutAttrs attrs{};
  attrs.p = static_cast<float>(p);
  attrs.seed = seed;
  attrs.scale_in_train = scale_in_train;

  // stream = null (default)
  int rc = ai::ops::dropout_run(tX, tY, pM, attrs, nullptr);
  cudaDeviceSynchronize();
  if (rc != 0) throw std::runtime_error("dropout_run failed");

  // D2H
  auto Y = py::array_t<float>({M, N});
  cudaMemcpy(Y.mutable_data(), dY.get(), sizeof(float) * count, cudaMemcpyDeviceToHost);

  if (dM) {
    auto Mhost = py::array_t<int32_t>({M, N});
    cudaMemcpy(Mhost.mutable_data(), dM.get(), sizeof(int32_t) * count, cudaMemcpyDeviceToHost);
    return py::make_tuple(std::move(Y), std::move(Mhost));
  } else {
    return Y;
  }
}

// backward(dY, mask, p, seed, scale_in_train) -> dX
static py::array dropout_backward(py::array dY_in, py::array mask_in,
                                  double p = 0.1, std::uint64_t seed = 0x1234,
                                  bool scale_in_train = true)
{
  auto dY = py::array_t<float,   py::array::c_style | py::array::forcecast>(dY_in);
  auto M  = py::array_t<int32_t, py::array::c_style | py::array::forcecast>(mask_in);

  if (dY.ndim()!=2 || M.ndim()!=2) throw std::runtime_error("dropout_backward: dY and mask must be 2D");
  if (dY.shape(0)!=M.shape(0) || dY.shape(1)!=M.shape(1)) throw std::runtime_error("shape mismatch");

  const int64_t R = dY.shape(0), C = dY.shape(1);
  const size_t  count = static_cast<size_t>(R) * static_cast<size_t>(C);

  // device alloc
  float* ddY_raw = nullptr;
  float* ddX_raw = nullptr;
  int32_t* dM_raw = nullptr;
  cudaMalloc(&ddY_raw, sizeof(float) * count);
  cudaMalloc(&ddX_raw, sizeof(float) * count);
  cudaMalloc(&dM_raw,  sizeof(int32_t) * count);

  cuda_unique_ptr<float> ddY(ddY_raw), ddX(ddX_raw);
  cuda_unique_ptr<int32_t> dM(dM_raw);

  // H2D
  cudaMemcpy(ddY.get(), dY.data(), sizeof(float) * count, cudaMemcpyHostToDevice);
  cudaMemcpy(dM.get(),  M.data(),  sizeof(int32_t) * count, cudaMemcpyHostToDevice);

  // wrap tensors
  Tensor tdY{ddY.get(), make_desc_2d_rowmajor(R, C, DType::F32), Device::CUDA, 0};
  Tensor tM {dM.get(),  make_desc_2d_rowmajor(R, C, DType::I32), Device::CUDA, 0};
  Tensor tdX{ddX.get(), make_desc_2d_rowmajor(R, C, DType::F32), Device::CUDA, 0};

  DropoutAttrs attrs{};
  attrs.p = static_cast<float>(p);
  attrs.seed = seed;
  attrs.scale_in_train = scale_in_train;

  int rc = ai::ops::dropout_backward_run(tdY, tM, tdX, attrs, nullptr);
  cudaDeviceSynchronize();
  if (rc != 0) throw std::runtime_error("dropout_backward_run failed");

  // D2H
  auto out = py::array_t<float>({R, C});
  cudaMemcpy(out.mutable_data(), ddX.get(), sizeof(float) * count, cudaMemcpyDeviceToHost);
  return out;
}

PYBIND11_MODULE(_ops_dropout, m) {
  m.doc() = "Standalone CUDA Dropout ops (forward/backward)";
  m.def("dropout", &dropout,
        py::arg("X"),
        py::arg("p") = 0.1,
        py::arg("return_mask") = py::bool_(false),
        py::arg("seed") = 0x1234,
        py::arg("scale_in_train") = true);

  m.def("dropout_backward", &dropout_backward,
        py::arg("dY"),
        py::arg("mask"),
        py::arg("p") = 0.1,
        py::arg("seed") = 0x1234,
        py::arg("scale_in_train") = true);
}
