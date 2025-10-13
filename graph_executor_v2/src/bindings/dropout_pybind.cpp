// src/bindings/dropout_pybind.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

#include "backends/cuda/ops/dropout/api.hpp"

namespace py = pybind11;
using namespace ai;

static Tensor make_tensor_rm(uintptr_t ptr_u64,
                             const std::vector<int64_t>& shape,
                             DType dtype, Device dev=Device::CUDA) {
  Tensor t;
  t.data = reinterpret_cast<void*>(ptr_u64);
  t.device = dev; t.device_index = 0;
  t.desc.dtype = dtype;
  t.desc.layout = Layout::RowMajor;
  t.desc.shape = shape;
  // contiguous row-major
  t.desc.stride.resize(shape.size());
  if (!shape.empty()) {
    t.desc.stride.back() = 1;
    for (int i = (int)shape.size()-2; i >= 0; --i) {
      t.desc.stride[i] = shape[i+1] * t.desc.stride[i+1];
    }
  }
  return t;
}

static void throw_if_bad(Status st, const char* where) {
  if (st != Status::Ok) {
    throw std::runtime_error(std::string("[_ops_dropout::") + where +
                             "] failed with Status=" +
                             std::to_string(static_cast<int>(st)));
  }
}

PYBIND11_MODULE(_ops_dropout, m) {
  m.attr("__package__") = "graph_executor_v2.ops";
  m.doc() = "CUDA dropout (stateless RNG; graph-capture safe)";

  // 재사용 타입
  py::module_ common = py::module_::import("graph_executor_v2.ops._ops_common");

  py::class_<DropoutAttrs>(m, "DropoutAttrs")
    .def(py::init<>())
    .def_readwrite("p", &DropoutAttrs::p)
    .def_readwrite("seed", &DropoutAttrs::seed)
    .def_readwrite("scale_in_train", &DropoutAttrs::scale_in_train)
    .def_readwrite("counter_base", &DropoutAttrs::counter_base);

  // forward(x, x_shape, y, y_shape, mask or None, attrs, stream)
  m.def("forward",
    [](uintptr_t x_ptr, const std::vector<int64_t>& x_shape,
       uintptr_t y_ptr, const std::vector<int64_t>& y_shape,
       py::object mask_ptr_obj,
       DropoutAttrs attrs, uintptr_t stream_ptr) {

      Tensor X = make_tensor_rm(x_ptr, x_shape, DType::F32);
      Tensor Y = make_tensor_rm(y_ptr, y_shape, DType::F32);
      Tensor* Mptr = nullptr;
      Tensor M;
      if (!mask_ptr_obj.is_none()) {
        auto mptr = mask_ptr_obj.cast<uintptr_t>();
        M = make_tensor_rm(mptr, x_shape, DType::I32); // mask shape == X/Y
        Mptr = &M;
      }
      StreamHandle stream = reinterpret_cast<StreamHandle>(stream_ptr);
      auto st = DropoutCudaLaunch(X, Y, Mptr, attrs, stream);
      throw_if_bad(st, "forward");
    },
    py::arg("x_ptr"), py::arg("x_shape"),
    py::arg("y_ptr"), py::arg("y_shape"),
    py::arg("mask_ptr") = py::none(),
    py::arg("attrs") = DropoutAttrs{},
    py::arg("stream") = (uintptr_t)0
  );

  // backward(dy, dy_shape, mask, mask_shape, dx, dx_shape, attrs, stream)
  m.def("backward",
    [](uintptr_t dy_ptr, const std::vector<int64_t>& dy_shape,
       uintptr_t mask_ptr, const std::vector<int64_t>& mask_shape,
       uintptr_t dx_ptr, const std::vector<int64_t>& dx_shape,
       DropoutAttrs attrs, uintptr_t stream_ptr) {

      Tensor dY = make_tensor_rm(dy_ptr, dy_shape, DType::F32);
      Tensor M  = make_tensor_rm(mask_ptr, mask_shape, DType::I32);
      Tensor dX = make_tensor_rm(dx_ptr, dx_shape, DType::F32);
      StreamHandle stream = reinterpret_cast<StreamHandle>(stream_ptr);
      auto st = DropoutCudaBackwardLaunch(dY, M, dX, attrs, stream);
      throw_if_bad(st, "backward");
    },
    py::arg("dy_ptr"), py::arg("dy_shape"),
    py::arg("mask_ptr"), py::arg("mask_shape"),
    py::arg("dx_ptr"), py::arg("dx_shape"),
    py::arg("attrs") = DropoutAttrs{},
    py::arg("stream") = (uintptr_t)0
  );
}
