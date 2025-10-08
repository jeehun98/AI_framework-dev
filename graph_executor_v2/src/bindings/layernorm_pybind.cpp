// src/bindings/layernorm_pybind.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

#include "backends/cuda/ops/layernorm/api.hpp"

namespace py = pybind11;
using namespace ai;

// ------------------------- helpers -------------------------
static Tensor make_tensor_nd(uintptr_t ptr_u64,
                             const std::vector<int64_t>& shape,
                             DType dtype = DType::F32,
                             Device dev  = Device::CUDA)
{
  Tensor t;
  t.data = reinterpret_cast<void*>(ptr_u64);
  t.device = dev;
  t.device_index = 0;

  t.desc.dtype  = dtype;
  t.desc.layout = Layout::RowMajor;
  t.desc.shape  = shape;

  // row-major contiguous strides
  const size_t R = shape.size();
  t.desc.stride.resize(R);
  if (R == 0) {
    t.desc.stride.clear();
  } else {
    t.desc.stride[R-1] = 1;
    for (int i = static_cast<int>(R) - 2; i >= 0; --i) {
      t.desc.stride[static_cast<size_t>(i)]
        = t.desc.stride[static_cast<size_t>(i+1)] * shape[static_cast<size_t>(i+1)];
    }
  }
  return t;
}

static void throw_if_bad(Status st, const char* where) {
  if (st != Status::Ok) {
    throw std::runtime_error(std::string(where) + " failed with Status=" +
                             std::to_string(static_cast<int>(st)));
  }
}

// ------------------------- module -------------------------
PYBIND11_MODULE(_ops_layernorm, m) {
  m.doc() = "Independent CUDA LayerNorm op bindings (forward/backward)";

  py::class_<LayerNormAttrs>(m, "LayerNormAttrs")
    .def(py::init<>())
    .def_readwrite("eps", &LayerNormAttrs::eps);

  // -------- Forward --------
  // X[M,N], (gamma[N] or None), (beta[N] or None) -> Y[M,N]
  // ws_fwd (optional): 현재 커널은 미사용. None만 허용.
  m.def("forward",
    [](uintptr_t x_ptr, const std::vector<int64_t>& x_shape,
       py::object gamma_ptr_or_none, const std::vector<int64_t>& gamma_shape,
       py::object beta_ptr_or_none,  const std::vector<int64_t>& beta_shape,
       uintptr_t y_ptr, const std::vector<int64_t>& y_shape,
       const LayerNormAttrs& attrs,
       uintptr_t stream_ptr,
       py::object /*ws_fwd_or_none*/) {

        if (x_shape.size() != 2 || y_shape.size() != 2)
          throw std::invalid_argument("LayerNorm.forward: X,Y must be rank-2 [M,N].");
        if (x_shape != y_shape)
          throw std::invalid_argument("LayerNorm.forward: X and Y shape must match [M,N].");

        const int64_t N = x_shape[1];

        Tensor X = make_tensor_nd(x_ptr, x_shape, DType::F32);
        Tensor Y = make_tensor_nd(y_ptr, y_shape, DType::F32);

        // gamma/beta: None 또는 [N]
        Tensor gT{}, bT{};
        const Tensor* gamma = nullptr;
        const Tensor* beta  = nullptr;

        if (!gamma_ptr_or_none.is_none()) {
          if (gamma_shape.size() != 1 || gamma_shape[0] != N)
            throw std::invalid_argument("LayerNorm.forward: gamma must be shape [N].");
          auto gptr = gamma_ptr_or_none.cast<uintptr_t>();
          gT = make_tensor_nd(gptr, {N}, DType::F32);
          gamma = &gT;
        }
        if (!beta_ptr_or_none.is_none()) {
          if (beta_shape.size() != 1 || beta_shape[0] != N)
            throw std::invalid_argument("LayerNorm.forward: beta must be shape [N].");
          auto bptr = beta_ptr_or_none.cast<uintptr_t>();
          bT = make_tensor_nd(bptr, {N}, DType::F32);
          beta = &bT;
        }

        StreamHandle s = reinterpret_cast<StreamHandle>(stream_ptr);
        // 현재 WS 미사용 → nullptr 전달
        auto st = LayerNormCudaLaunch(X, gamma, beta, Y, attrs, s, /*ws_fwd=*/nullptr);
        throw_if_bad(st, "LayerNormCudaLaunch");
      },
    py::arg("x_ptr"),    py::arg("x_shape"),
    py::arg("gamma_ptr") = py::none(), py::arg("gamma_shape") = std::vector<int64_t>{},
    py::arg("beta_ptr")  = py::none(), py::arg("beta_shape")  = std::vector<int64_t>{},
    py::arg("y_ptr"),    py::arg("y_shape"),
    py::arg("attrs"),
    py::arg("stream") = static_cast<uintptr_t>(0),
    py::arg("ws_fwd") = py::none()  // ← 추가
  );

  // -------- Backward --------
  // (X[M,N], gamma[N]? , dY[M,N]) -> dX[M,N] , (dgamma[N]?, dbeta[N]?)
  // ws_bwd (optional): 현재 커널은 미사용. None만 허용.
  m.def("backward",
    [](uintptr_t x_ptr,  const std::vector<int64_t>& x_shape,
       py::object gamma_ptr_or_none, const std::vector<int64_t>& gamma_shape,
       uintptr_t dy_ptr, const std::vector<int64_t>& dy_shape,
       uintptr_t dx_ptr, const std::vector<int64_t>& dx_shape,
       py::object dgamma_ptr_or_none, const std::vector<int64_t>& dgamma_shape,
       py::object dbeta_ptr_or_none,  const std::vector<int64_t>& dbeta_shape,
       const LayerNormAttrs& attrs,
       uintptr_t stream_ptr,
       py::object /*ws_bwd_or_none*/) {

        if (x_shape.size() != 2 || dy_shape.size() != 2 || dx_shape.size() != 2)
          throw std::invalid_argument("LayerNorm.backward: X,dY,dX must be rank-2 [M,N].");
        if (x_shape != dy_shape || x_shape != dx_shape)
          throw std::invalid_argument("LayerNorm.backward: shapes of X,dY,dX must match [M,N].");

        const int64_t N = x_shape[1];

        Tensor X  = make_tensor_nd(x_ptr,  x_shape,  DType::F32);
        Tensor dY = make_tensor_nd(dy_ptr, dy_shape, DType::F32);
        Tensor dX = make_tensor_nd(dx_ptr, dx_shape, DType::F32);

        // optional gamma
        Tensor gT{}; const Tensor* gamma = nullptr;
        if (!gamma_ptr_or_none.is_none()) {
          if (gamma_shape.size() != 1 || gamma_shape[0] != N)
            throw std::invalid_argument("LayerNorm.backward: gamma must be shape [N].");
          auto gptr = gamma_ptr_or_none.cast<uintptr_t>();
          gT = make_tensor_nd(gptr, {N}, DType::F32);
          gamma = &gT;
        }

        // optional outputs dgamma/dbeta
        Tensor dgammaT{}, dbetaT{};
        Tensor* dgamma = nullptr;
        Tensor* dbeta  = nullptr;

        if (!dgamma_ptr_or_none.is_none()) {
          if (dgamma_shape.size() != 1 || dgamma_shape[0] != N)
            throw std::invalid_argument("LayerNorm.backward: dgamma must be shape [N].");
          auto goptr = dgamma_ptr_or_none.cast<uintptr_t>();
          dgammaT = make_tensor_nd(goptr, {N}, DType::F32);
          dgamma  = &dgammaT;
        }
        if (!dbeta_ptr_or_none.is_none()) {
          if (dbeta_shape.size() != 1 || dbeta_shape[0] != N)
            throw std::invalid_argument("LayerNorm.backward: dbeta must be shape [N].");
          auto boptr = dbeta_ptr_or_none.cast<uintptr_t>();
          dbetaT = make_tensor_nd(boptr, {N}, DType::F32);
          dbeta  = &dbetaT;
        }

        StreamHandle s = reinterpret_cast<StreamHandle>(stream_ptr);
        // 현재 WS 미사용 → nullptr 전달
        auto st = LayerNormCudaBackwardLaunch(X, gamma, dY, dX, dgamma, dbeta, attrs, s, /*ws_bwd=*/nullptr);
        throw_if_bad(st, "LayerNormCudaBackwardLaunch");
      },
    py::arg("x_ptr"),     py::arg("x_shape"),
    py::arg("gamma_ptr") = py::none(), py::arg("gamma_shape") = std::vector<int64_t>{},
    py::arg("dy_ptr"),    py::arg("dy_shape"),
    py::arg("dx_ptr"),    py::arg("dx_shape"),
    py::arg("dgamma_ptr") = py::none(), py::arg("dgamma_shape") = std::vector<int64_t>{},
    py::arg("dbeta_ptr")  = py::none(), py::arg("dbeta_shape")  = std::vector<int64_t>{},
    py::arg("attrs"),
    py::arg("stream") = static_cast<uintptr_t>(0),
    py::arg("ws_bwd") = py::none()  // ← 추가
  );
}
