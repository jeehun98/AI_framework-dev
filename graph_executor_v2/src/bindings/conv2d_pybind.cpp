#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

#include "backends/cuda/ops/conv2d/api.hpp"

namespace py = pybind11;
using namespace ai;

// 편의: row-major 4D/1D 텐서 생성기 (stride 자동: contiguous)
static Tensor make_tensor_4d(uintptr_t ptr_u64,
                             const std::vector<int64_t>& shape, // [N,C,H,W]
                             DType dtype = DType::F32,
                             Device dev  = Device::CUDA) {
  if (shape.size() != 4) throw std::invalid_argument("shape must be 4D [N,C,H,W]");
  Tensor t;
  t.data = reinterpret_cast<void*>(ptr_u64);
  t.device = dev;
  t.device_index = 0;
  t.desc.dtype = dtype;
  t.desc.layout = Layout::RowMajor;
  t.desc.shape = shape;
  // contiguous row-major stride
  t.desc.stride.resize(4);
  t.desc.stride[3] = 1;
  t.desc.stride[2] = shape[3] * t.desc.stride[3];
  t.desc.stride[1] = shape[2] * t.desc.stride[2];
  t.desc.stride[0] = shape[1] * t.desc.stride[1];
  return t;
}

static Tensor make_tensor_1d(uintptr_t ptr_u64,
                             int64_t len,
                             DType dtype = DType::F32,
                             Device dev  = Device::CUDA) {
  Tensor t;
  t.data = reinterpret_cast<void*>(ptr_u64);
  t.device = dev;
  t.device_index = 0;
  t.desc.dtype = dtype;
  t.desc.layout = Layout::RowMajor;
  t.desc.shape  = { len };
  t.desc.stride = { 1 };
  return t;
}

// 예외 변환: Status != Ok -> Python RuntimeError
static void throw_if_bad(Status st, const char* where) {
  if (st != Status::Ok) {
    throw std::runtime_error(std::string(where) + " failed with Status=" + std::to_string(static_cast<int>(st)));
  }
}

PYBIND11_MODULE(_ops_conv2d, m) {
  m.doc() = "Independent conv2d CUDA ops binding (standalone shim compatible)";

  // Conv2DAttrs
  py::class_<Conv2DAttrs>(m, "Conv2DAttrs")
    .def(py::init<>())
    .def_readwrite("stride_h", &Conv2DAttrs::stride_h)
    .def_readwrite("stride_w", &Conv2DAttrs::stride_w)
    .def_readwrite("pad_h",    &Conv2DAttrs::pad_h)
    .def_readwrite("pad_w",    &Conv2DAttrs::pad_w)
    .def_readwrite("dil_h",    &Conv2DAttrs::dil_h)
    .def_readwrite("dil_w",    &Conv2DAttrs::dil_w)
    .def_readwrite("groups",   &Conv2DAttrs::groups);

  // forward
  m.def("forward",
    [](uintptr_t x_ptr, const std::vector<int64_t>& x_shape,       // [N,Cin,H,W]
       uintptr_t w_ptr, const std::vector<int64_t>& w_shape,       // [Cout,Cin,Kh,Kw]
       uintptr_t y_ptr, const std::vector<int64_t>& y_shape,       // [N,Cout,Hout,Wout]
       py::object b_ptr_obj,                                       // int or None (bias)
       Conv2DAttrs attrs,
       uintptr_t stream_ptr) {

        Tensor X = make_tensor_4d(x_ptr, x_shape);
        Tensor W; {
          if (w_shape.size() != 4) throw std::invalid_argument("W shape must be 4D [Cout,Cin,Kh,Kw]");
          W = make_tensor_4d(w_ptr, {w_shape[0], w_shape[1], w_shape[2], w_shape[3]});
        }
        Tensor Y = make_tensor_4d(y_ptr, y_shape);

        const Tensor* Bptr = nullptr;
        Tensor B;
        if (!b_ptr_obj.is_none()) {
          auto b_ptr = b_ptr_obj.cast<uintptr_t>();
          // bias is [Cout]
          B = make_tensor_1d(b_ptr, w_shape.at(0));
          Bptr = &B;
        }

        StreamHandle stream = reinterpret_cast<StreamHandle>(stream_ptr);
        auto st = Conv2DCudaLaunch(X, W, Bptr, Y, attrs, stream);
        throw_if_bad(st, "Conv2DCudaLaunch");
      },
    py::arg("x_ptr"), py::arg("x_shape"),
    py::arg("w_ptr"), py::arg("w_shape"),
    py::arg("y_ptr"), py::arg("y_shape"),
    py::arg("bias_ptr") = py::none(),
    py::arg("attrs")    = Conv2DAttrs{},
    py::arg("stream")   = static_cast<uintptr_t>(0)
  );

  // backward
  m.def("backward",
    [](uintptr_t x_ptr, const std::vector<int64_t>& x_shape,       // [N,Cin,H,W]
       uintptr_t w_ptr, const std::vector<int64_t>& w_shape,       // [Cout,Cin,Kh,Kw]
       uintptr_t dy_ptr, const std::vector<int64_t>& dy_shape,     // [N,Cout,Hout,Wout]
       py::object dw_ptr_obj,                                      // int or None
       py::object db_ptr_obj,                                      // int or None
       py::object dx_ptr_obj,                                      // int or None
       Conv2DAttrs attrs,
       uintptr_t stream_ptr) {

        Tensor X = make_tensor_4d(x_ptr, x_shape);
        Tensor W = make_tensor_4d(w_ptr, w_shape);
        Tensor dY= make_tensor_4d(dy_ptr, dy_shape);

        Tensor *dW=nullptr, *dB=nullptr, *dX=nullptr;
        Tensor dW_t, dB_t, dX_t;

        if (!dw_ptr_obj.is_none()) {
          auto dw_ptr = dw_ptr_obj.cast<uintptr_t>();
          dW_t = make_tensor_4d(dw_ptr, w_shape);
          dW = &dW_t;
        }
        if (!db_ptr_obj.is_none()) {
          auto db_ptr = db_ptr_obj.cast<uintptr_t>();
          dB_t = make_tensor_1d(db_ptr, w_shape.at(0)); // Cout
          dB = &dB_t;
        }
        if (!dx_ptr_obj.is_none()) {
          auto dx_ptr = dx_ptr_obj.cast<uintptr_t>();
          dX_t = make_tensor_4d(dx_ptr, x_shape);
          dX = &dX_t;
        }

        StreamHandle stream = reinterpret_cast<StreamHandle>(stream_ptr);
        auto st = Conv2DCudaBackwardLaunch(X, W, dY, dW, dB, dX, attrs, stream);
        throw_if_bad(st, "Conv2DCudaBackwardLaunch");
      },
    py::arg("x_ptr"), py::arg("x_shape"),
    py::arg("w_ptr"), py::arg("w_shape"),
    py::arg("dy_ptr"), py::arg("dy_shape"),
    py::arg("dw_ptr") = py::none(),
    py::arg("db_ptr") = py::none(),
    py::arg("dx_ptr") = py::none(),
    py::arg("attrs")  = Conv2DAttrs{},
    py::arg("stream") = static_cast<uintptr_t>(0)
  );
}
