#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

#include "backends/cuda/ops/pool2d/api.hpp"

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

  // contiguous row-major strides
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
    throw std::runtime_error(std::string(where) + " failed with Status=" + std::to_string(static_cast<int>(st)));
  }
}

// ------------------------- module -------------------------
PYBIND11_MODULE(_ops_pool2d, m) {
  m.doc() = "Independent pool2d CUDA op bindings (MaxPool/AvgPool)";

  // Pool2DAttrs
  py::class_<Pool2DAttrs>(m, "Pool2DAttrs")
    .def(py::init<>())
    .def_readwrite("kH", &Pool2DAttrs::kH)
    .def_readwrite("kW", &Pool2DAttrs::kW)
    .def_readwrite("sH", &Pool2DAttrs::sH)
    .def_readwrite("sW", &Pool2DAttrs::sW)
    .def_readwrite("pH", &Pool2DAttrs::pH)
    .def_readwrite("pW", &Pool2DAttrs::pW)
    .def_readwrite("dH", &Pool2DAttrs::dH)
    .def_readwrite("dW", &Pool2DAttrs::dW)
    .def_readwrite("ceil_mode", &Pool2DAttrs::ceil_mode)
    .def_readwrite("count_include_pad", &Pool2DAttrs::count_include_pad);

  // -------- MaxPool2D Forward --------
  // X[N,C,H,W] -> Y[N,C,Ho,Wo], (optional) Indices[N,C,Ho,Wo] (int32)
  m.def("maxpool2d_forward",
    [](uintptr_t x_ptr, const std::vector<int64_t>& x_shape,
       uintptr_t y_ptr, const std::vector<int64_t>& y_shape,
       py::object indices_ptr_or_none,  // None or uintptr_t
       const Pool2DAttrs& attrs,
       uintptr_t stream_ptr) {

        if (x_shape.size() != 4 || y_shape.size() != 4) {
          throw std::invalid_argument("X and Y must be rank-4 (NCHW).");
        }

        Tensor X = make_tensor_nd(x_ptr, x_shape, DType::F32);
        Tensor Y = make_tensor_nd(y_ptr, y_shape, DType::F32);

        Tensor* Indices = nullptr;
        Tensor Ind;
        if (!indices_ptr_or_none.is_none()) {
          auto ind_ptr = indices_ptr_or_none.cast<uintptr_t>();
          Ind = make_tensor_nd(ind_ptr, y_shape, DType::I32); // same shape as Y
          Indices = &Ind;
        }

        StreamHandle s = reinterpret_cast<StreamHandle>(stream_ptr);
        auto st = MaxPool2DCudaLaunch(X, Y, Indices, attrs, s);
        throw_if_bad(st, "MaxPool2DCudaLaunch");
      },
    py::arg("x_ptr"), py::arg("x_shape"),
    py::arg("y_ptr"), py::arg("y_shape"),
    py::arg("indices_ptr") = py::none(),
    py::arg("attrs"),
    py::arg("stream") = static_cast<uintptr_t>(0)
  );

  // -------- MaxPool2D Backward --------
  // dY[N,C,Ho,Wo], Indices[N,C,Ho,Wo](int32) -> dX[N,C,H,W]
  m.def("maxpool2d_backward",
    [](uintptr_t dy_ptr, const std::vector<int64_t>& dy_shape,
       uintptr_t indices_ptr, const std::vector<int64_t>& indices_shape,
       uintptr_t dx_ptr, const std::vector<int64_t>& dx_shape,
       const Pool2DAttrs& attrs,
       uintptr_t stream_ptr) {

        if (dy_shape.size() != 4 || indices_shape.size() != 4 || dx_shape.size() != 4) {
          throw std::invalid_argument("dY, Indices, dX must be rank-4 (NCHW).");
        }
        if (dy_shape != indices_shape) {
          throw std::invalid_argument("Indices shape must equal dY(shape==Y).");
        }

        Tensor dY = make_tensor_nd(dy_ptr, dy_shape, DType::F32);
        Tensor Ind = make_tensor_nd(indices_ptr, indices_shape, DType::I32);
        Tensor dX = make_tensor_nd(dx_ptr, dx_shape, DType::F32);

        StreamHandle s = reinterpret_cast<StreamHandle>(stream_ptr);
        auto st = MaxPool2DBackwardCudaLaunch(dY, Ind, dX, attrs, s);
        throw_if_bad(st, "MaxPool2DBackwardCudaLaunch");
      },
    py::arg("dy_ptr"), py::arg("dy_shape"),
    py::arg("indices_ptr"), py::arg("indices_shape"),
    py::arg("dx_ptr"), py::arg("dx_shape"),
    py::arg("attrs"),
    py::arg("stream") = static_cast<uintptr_t>(0)
  );

  // -------- AvgPool2D Forward --------
  m.def("avgpool2d_forward",
    [](uintptr_t x_ptr, const std::vector<int64_t>& x_shape,
       uintptr_t y_ptr, const std::vector<int64_t>& y_shape,
       const Pool2DAttrs& attrs,
       uintptr_t stream_ptr) {

        if (x_shape.size() != 4 || y_shape.size() != 4) {
          throw std::invalid_argument("X and Y must be rank-4 (NCHW).");
        }

        Tensor X = make_tensor_nd(x_ptr, x_shape, DType::F32);
        Tensor Y = make_tensor_nd(y_ptr, y_shape, DType::F32);

        StreamHandle s = reinterpret_cast<StreamHandle>(stream_ptr);
        auto st = AvgPool2DCudaLaunch(X, Y, attrs, s);
        throw_if_bad(st, "AvgPool2DCudaLaunch");
      },
    py::arg("x_ptr"), py::arg("x_shape"),
    py::arg("y_ptr"), py::arg("y_shape"),
    py::arg("attrs"),
    py::arg("stream") = static_cast<uintptr_t>(0)
  );

  // -------- AvgPool2D Backward --------
  m.def("avgpool2d_backward",
    [](uintptr_t dy_ptr, const std::vector<int64_t>& dy_shape,
       uintptr_t dx_ptr, const std::vector<int64_t>& dx_shape,
       const Pool2DAttrs& attrs,
       uintptr_t stream_ptr) {

        if (dy_shape.size() != 4 || dx_shape.size() != 4) {
          throw std::invalid_argument("dY and dX must be rank-4 (NCHW).");
        }

        Tensor dY = make_tensor_nd(dy_ptr, dy_shape, DType::F32);
        Tensor dX = make_tensor_nd(dx_ptr, dx_shape, DType::F32);

        StreamHandle s = reinterpret_cast<StreamHandle>(stream_ptr);
        auto st = AvgPool2DBackwardCudaLaunch(dY, dX, attrs, s);
        throw_if_bad(st, "AvgPool2DBackwardCudaLaunch");
      },
    py::arg("dy_ptr"), py::arg("dy_shape"),
    py::arg("dx_ptr"), py::arg("dx_shape"),
    py::arg("attrs"),
    py::arg("stream") = static_cast<uintptr_t>(0)
  );
}
