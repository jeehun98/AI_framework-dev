// src/bindings/dropout_pybind.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#include "backends/cuda/ops/dropout/api.hpp"

namespace py = pybind11;
using namespace ai;

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
  const size_t R = shape.size();
  t.desc.stride.resize(R);
  if (R) {
    t.desc.stride[R-1] = 1;
    for (int i=(int)R-2;i>=0;--i)
      t.desc.stride[(size_t)i] = t.desc.stride[(size_t)i+1]*shape[(size_t)i+1];
  }
  return t;
}

static void throw_if_bad(Status st, const char* where) {
  if (st != Status::Ok) {
    throw std::runtime_error(std::string(where) + " failed with Status=" +
                             std::to_string(static_cast<int>(st)));
  }
}

PYBIND11_MODULE(_ops_dropout, m) {
  m.doc() = "Independent CUDA Dropout (stateless RNG, capture-safe)";

  py::class_<DropoutAttrs>(m, "DropoutAttrs")
    .def(py::init<>())
    .def_readwrite("p", &DropoutAttrs::p)
    .def_readwrite("seed", &DropoutAttrs::seed)
    .def_readwrite("scale_in_train", &DropoutAttrs::scale_in_train)
    .def_readwrite("counter_base", &DropoutAttrs::counter_base);

  // forward
  m.def("forward",
    [](uintptr_t x_ptr, const std::vector<int64_t>& x_shape,
       uintptr_t y_ptr, const std::vector<int64_t>& y_shape,
       py::object mask_ptr_or_none, const std::vector<int64_t>& mask_shape,
       const DropoutAttrs& attrs, uintptr_t stream_ptr) {

        if (x_shape.size()!=2 || y_shape.size()!=2)
          throw std::invalid_argument("X and Y must be rank-2 [M,N]");
        if (x_shape != y_shape)
          throw std::invalid_argument("X and Y shapes must match");

        Tensor X = make_tensor_nd(x_ptr, x_shape, DType::F32);
        Tensor Y = make_tensor_nd(y_ptr, y_shape, DType::F32);

        Tensor maskT{}; Tensor* mask = nullptr;
        if (!mask_ptr_or_none.is_none()) {
          if (mask_shape.size()!=2 || mask_shape!=y_shape)
            throw std::invalid_argument("mask must be [M,N] (int32) matching Y shape");
          auto mptr = mask_ptr_or_none.cast<uintptr_t>();
          maskT = make_tensor_nd(mptr, mask_shape, DType::I32);
          mask = &maskT;
        }

        StreamHandle s = reinterpret_cast<StreamHandle>(stream_ptr);
        auto st = DropoutCudaLaunch(X, Y, mask, attrs, s);
        throw_if_bad(st, "DropoutCudaLaunch");
      },
    py::arg("x_ptr"), py::arg("x_shape"),
    py::arg("y_ptr"), py::arg("y_shape"),
    py::arg("mask_ptr") = py::none(), py::arg("mask_shape") = std::vector<int64_t>{},
    py::arg("attrs"),
    py::arg("stream") = static_cast<uintptr_t>(0)
  );

  // backward
  m.def("backward",
    [](uintptr_t dy_ptr, const std::vector<int64_t>& dy_shape,
       uintptr_t mask_ptr, const std::vector<int64_t>& mask_shape,
       uintptr_t dx_ptr, const std::vector<int64_t>& dx_shape,
       const DropoutAttrs& attrs, uintptr_t stream_ptr) {

        if (dy_shape.size()!=2 || dx_shape.size()!=2 || mask_shape.size()!=2)
          throw std::invalid_argument("dY, dX, mask must be rank-2 [M,N]");
        if (dy_shape!=dx_shape || dy_shape!=mask_shape)
          throw std::invalid_argument("dY, dX, mask shapes must match [M,N]");

        Tensor dY = make_tensor_nd(dy_ptr,   dy_shape, DType::F32);
        Tensor M  = make_tensor_nd(mask_ptr, mask_shape, DType::I32);
        Tensor dX = make_tensor_nd(dx_ptr,   dx_shape, DType::F32);

        StreamHandle s = reinterpret_cast<StreamHandle>(stream_ptr);
        auto st = DropoutCudaBackwardLaunch(dY, M, dX, attrs, s);
        throw_if_bad(st, "DropoutCudaBackwardLaunch");
      },
    py::arg("dy_ptr"),   py::arg("dy_shape"),
    py::arg("mask_ptr"), py::arg("mask_shape"),
    py::arg("dx_ptr"),   py::arg("dx_shape"),
    py::arg("attrs"),
    py::arg("stream") = static_cast<uintptr_t>(0)
  );
}
