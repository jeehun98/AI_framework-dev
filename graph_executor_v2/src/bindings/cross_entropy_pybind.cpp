// src/bindings/cross_entropy_pybind.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>


#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

#include "backends/cuda/ops/cross_entropy/api.hpp"

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
PYBIND11_MODULE(_ops_cross_entropy, m) {
  m.doc() = "Independent CUDA Cross-Entropy (forward/backward) — int32 targets only";

  py::enum_<Reduction>(m, "Reduction")
    .value("None_", Reduction::None)
    .value("Mean",  Reduction::Mean)
    .value("Sum",   Reduction::Sum)
    .export_values();

  py::class_<CrossEntropyAttrs>(m, "CrossEntropyAttrs")
    .def(py::init<>())
    .def_readwrite("from_logits", &CrossEntropyAttrs::from_logits)
    .def_readwrite("reduction",   &CrossEntropyAttrs::reduction)
    .def_readwrite("ignore_index",&CrossEntropyAttrs::ignore_index)
    .def_readwrite("eps",         &CrossEntropyAttrs::eps)
    .def_readwrite("ls_eps",      &CrossEntropyAttrs::ls_eps);

  // -------- Forward --------
  // X[M,N], target[M](int32) -> loss
  m.def("forward",
    [](uintptr_t x_ptr, const std::vector<int64_t>& x_shape,
       uintptr_t t_ptr, const std::vector<int64_t>& t_shape,
       uintptr_t loss_ptr, const std::vector<int64_t>& loss_shape,
       const CrossEntropyAttrs& attrs,
       uintptr_t stream_ptr) {

        if (x_shape.size() != 2) throw std::invalid_argument("X must be [M,N]");
        if (t_shape.size() != 1) throw std::invalid_argument("target must be [M]");
        const int64_t M = x_shape[0];
        if (t_shape[0] != M) throw std::invalid_argument("target length must equal M");

        // loss shape 검증
        if (attrs.reduction == Reduction::None) {
          if (!(loss_shape.size()==1 && loss_shape[0]==M))
            throw std::invalid_argument("loss must be [M] when reduction=None");
        } else {
          if (!(loss_shape.size()==1 && loss_shape[0]==1))
            throw std::invalid_argument("loss must be [1] when reduction=Mean/Sum");
        }

        Tensor X    = make_tensor_nd(x_ptr,   x_shape,   DType::F32);
        Tensor T    = make_tensor_nd(t_ptr,   t_shape,   DType::I32); // I32 고정
        Tensor Loss = make_tensor_nd(loss_ptr,loss_shape,DType::F32);

        StreamHandle s = reinterpret_cast<StreamHandle>(stream_ptr);
        auto st = CrossEntropyCudaLaunch(X, T, Loss, attrs, s);
        throw_if_bad(st, "CrossEntropyCudaLaunch");
      },
    py::arg("x_ptr"), py::arg("x_shape"),
    py::arg("t_ptr"), py::arg("t_shape"),
    py::arg("loss_ptr"), py::arg("loss_shape"),
    py::arg("attrs"),
    py::arg("stream") = static_cast<uintptr_t>(0)
  );

  // -------- Backward --------
  // dX[M,N]
  m.def("backward",
    [](uintptr_t x_ptr, const std::vector<int64_t>& x_shape,
       uintptr_t t_ptr, const std::vector<int64_t>& t_shape,
       uintptr_t dx_ptr, const std::vector<int64_t>& dx_shape,
       const CrossEntropyAttrs& attrs,
       uintptr_t stream_ptr) {

        if (x_shape.size() != 2 || dx_shape.size() != 2)
          throw std::invalid_argument("X and dX must be [M,N]");
        if (x_shape != dx_shape)
          throw std::invalid_argument("X and dX shapes must match");
        if (t_shape.size() != 1 || t_shape[0] != x_shape[0])
          throw std::invalid_argument("target must be [M]");

        Tensor X  = make_tensor_nd(x_ptr,  x_shape,  DType::F32);
        Tensor T  = make_tensor_nd(t_ptr,  t_shape,  DType::I32); // I32 고정
        Tensor dX = make_tensor_nd(dx_ptr, dx_shape, DType::F32);

        StreamHandle s = reinterpret_cast<StreamHandle>(stream_ptr);
        auto st = CrossEntropyCudaBackwardLaunch(X, T, dX, attrs, s);
        throw_if_bad(st, "CrossEntropyCudaBackwardLaunch");
      },
    py::arg("x_ptr"), py::arg("x_shape"),
    py::arg("t_ptr"), py::arg("t_shape"),
    py::arg("dx_ptr"), py::arg("dx_shape"),
    py::arg("attrs"),
    py::arg("stream") = static_cast<uintptr_t>(0)
  );
}
