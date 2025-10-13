// src/bindings/pool2d_pybind.cpp
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
  m.doc() = "Independent pool2d CUDA op bindings (MaxPool/AvgPool) with capture-safe workspaces";

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

  // ---------------- MaxPool Workspaces (lightweight wrappers) ----------------
  py::class_<MaxPool2DWorkspaceFwd>(m, "MaxPool2DWorkspaceFwd")
    .def(py::init<>())
    .def_readwrite("indices", &MaxPool2DWorkspaceFwd::indices);

  py::class_<MaxPool2DWorkspaceBwd>(m, "MaxPool2DWorkspaceBwd")
    .def(py::init<>())
    .def_readwrite("indices", &MaxPool2DWorkspaceBwd::indices)
    .def_readwrite("scratch", &MaxPool2DWorkspaceBwd::scratch);

  // AvgPool WS (현재는 선택/미사용 경로)
  py::class_<AvgPool2DWorkspaceFwd>(m, "AvgPool2DWorkspaceFwd")
    .def(py::init<>())
    .def_readwrite("scratch", &AvgPool2DWorkspaceFwd::scratch);

  py::class_<AvgPool2DWorkspaceBwd>(m, "AvgPool2DWorkspaceBwd")
    .def(py::init<>())
    .def_readwrite("scratch", &AvgPool2DWorkspaceBwd::scratch);

  // -------- MaxPool2D Forward --------
  // X[N,C,H,W] -> Y[N,C,Ho,Wo], (optional) Indices[N,C,Ho,Wo] (int32)
  // ws_indices_ptr: 그래프 캡처용 외부 인덱스 버퍼(선택). 제공 시 [N,C,Ho,Wo] int32 크기로 호출자 책임.
  m.def("maxpool2d_forward",
    [](uintptr_t x_ptr, const std::vector<int64_t>& x_shape,
       uintptr_t y_ptr, const std::vector<int64_t>& y_shape,
       py::object indices_ptr_or_none,          // None or uintptr_t
       py::object ws_indices_ptr_or_none,       // None or uintptr_t
       const Pool2DAttrs& attrs,
       uintptr_t stream_ptr) {

        if (x_shape.size() != 4 || y_shape.size() != 4) {
          throw std::invalid_argument("X and Y must be rank-4 (NCHW).");
        }

        Tensor X = make_tensor_nd(x_ptr, x_shape, DType::F32);
        Tensor Y = make_tensor_nd(y_ptr, y_shape, DType::F32);

        // Optional Indices tensor
        Tensor* Indices = nullptr;
        Tensor Ind;
        if (!indices_ptr_or_none.is_none()) {
          auto ind_ptr = indices_ptr_or_none.cast<uintptr_t>();
          Ind = make_tensor_nd(ind_ptr, y_shape, DType::I32); // same shape as Y
          Indices = &Ind;
        }

        // Optional WS(indices)
        MaxPool2DWorkspaceFwd ws_local{};
        const MaxPool2DWorkspaceFwd* ws = nullptr;
        if (!ws_indices_ptr_or_none.is_none()) {
          ws_local.indices = reinterpret_cast<int32_t*>(ws_indices_ptr_or_none.cast<uintptr_t>());
          ws = &ws_local;
        }

        StreamHandle s = reinterpret_cast<StreamHandle>(stream_ptr);
        auto st = MaxPool2DCudaLaunch(X, Y, Indices, attrs, s, /*ws_fwd=*/ws);
        throw_if_bad(st, "MaxPool2DCudaLaunch");
      },
    py::arg("x_ptr"), py::arg("x_shape"),
    py::arg("y_ptr"), py::arg("y_shape"),
    py::arg("indices_ptr") = py::none(),
    py::arg("ws_indices_ptr") = py::none(),
    py::arg("attrs"),
    py::arg("stream") = static_cast<uintptr_t>(0)
  );

  // -------- MaxPool2D Backward --------
  // dY[N,C,Ho,Wo], (optional) Indices[N,C,Ho,Wo](int32) OR ws_indices_ptr -> dX[N,C,H,W]
  m.def("maxpool2d_backward",
    [](uintptr_t dy_ptr, const std::vector<int64_t>& dy_shape,
       py::object indices_ptr_or_none,          // None or uintptr_t
       uintptr_t dx_ptr, const std::vector<int64_t>& dx_shape,
       py::object ws_indices_ptr_or_none,       // None or uintptr_t
       py::object ws_scratch_ptr_or_none,       // None or uintptr_t (옵션)
       const Pool2DAttrs& attrs,
       uintptr_t stream_ptr) {

        if (dy_shape.size() != 4 || dx_shape.size() != 4) {
          throw std::invalid_argument("dY and dX must be rank-4 (NCHW).");
        }

        Tensor dY = make_tensor_nd(dy_ptr, dy_shape, DType::F32);
        Tensor dX = make_tensor_nd(dx_ptr, dx_shape, DType::F32);

        // Optional Indices tensor
        const Tensor* Indices = nullptr;
        Tensor Ind;
        if (!indices_ptr_or_none.is_none()) {
          auto ind_ptr = indices_ptr_or_none.cast<uintptr_t>();
          // shape는 dY와 동일
          Ind = make_tensor_nd(ind_ptr, dy_shape, DType::I32);
          Indices = &Ind;
        }

        // WS 구성
        MaxPool2DWorkspaceBwd ws_local{};
        const MaxPool2DWorkspaceBwd* ws = nullptr;

        if (!ws_indices_ptr_or_none.is_none()) {
          ws_local.indices = reinterpret_cast<int32_t*>(ws_indices_ptr_or_none.cast<uintptr_t>());
          ws = &ws_local;
        }
        if (!ws_scratch_ptr_or_none.is_none()) {
          ws_local.scratch = reinterpret_cast<float*>(ws_scratch_ptr_or_none.cast<uintptr_t>());
          ws = &ws_local;
        }

        // 인덱스 소스가 전혀 없으면 에러(현재 커널 재계산 미지원)
        if (Indices == nullptr && (ws == nullptr || ws->indices == nullptr)) {
          throw std::invalid_argument("MaxPool2D backward requires either Indices tensor or ws_indices_ptr.");
        }

        StreamHandle s = reinterpret_cast<StreamHandle>(stream_ptr);
        auto st = MaxPool2DBackwardCudaLaunch(dY, dX, Indices, attrs, s, /*ws_bwd=*/ws);
        throw_if_bad(st, "MaxPool2DBackwardCudaLaunch");
      },
    py::arg("dy_ptr"), py::arg("dy_shape"),
    py::arg("indices_ptr") = py::none(),
    py::arg("dx_ptr"), py::arg("dx_shape"),
    py::arg("ws_indices_ptr") = py::none(),
    py::arg("ws_scratch_ptr") = py::none(),
    py::arg("attrs"),
    py::arg("stream") = static_cast<uintptr_t>(0)
  );

  // -------- AvgPool2D Forward --------
  m.def("avgpool2d_forward",
    [](uintptr_t x_ptr, const std::vector<int64_t>& x_shape,
       uintptr_t y_ptr, const std::vector<int64_t>& y_shape,
       py::object ws_scratch_ptr_or_none,       // None or uintptr_t (현재 미사용)
       const Pool2DAttrs& attrs,
       uintptr_t stream_ptr) {

        if (x_shape.size() != 4 || y_shape.size() != 4) {
          throw std::invalid_argument("X and Y must be rank-4 (NCHW).");
        }

        Tensor X = make_tensor_nd(x_ptr, x_shape, DType::F32);
        Tensor Y = make_tensor_nd(y_ptr, y_shape, DType::F32);

        AvgPool2DWorkspaceFwd ws_local{};
        const AvgPool2DWorkspaceFwd* ws = nullptr;
        if (!ws_scratch_ptr_or_none.is_none()) {
          ws_local.scratch = reinterpret_cast<float*>(ws_scratch_ptr_or_none.cast<uintptr_t>());
          ws = &ws_local;
        }

        StreamHandle s = reinterpret_cast<StreamHandle>(stream_ptr);
        auto st = AvgPool2DCudaLaunch(X, Y, attrs, s, /*ws_fwd=*/ws);
        throw_if_bad(st, "AvgPool2DCudaLaunch");
      },
    py::arg("x_ptr"), py::arg("x_shape"),
    py::arg("y_ptr"), py::arg("y_shape"),
    py::arg("ws_scratch_ptr") = py::none(),
    py::arg("attrs"),
    py::arg("stream") = static_cast<uintptr_t>(0)
  );

  // -------- AvgPool2D Backward --------
  m.def("avgpool2d_backward",
    [](uintptr_t dy_ptr, const std::vector<int64_t>& dy_shape,
       uintptr_t dx_ptr, const std::vector<int64_t>& dx_shape,
       py::object ws_scratch_ptr_or_none,       // None or uintptr_t (현재 미사용)
       const Pool2DAttrs& attrs,
       uintptr_t stream_ptr) {

        if (dy_shape.size() != 4 || dx_shape.size() != 4) {
          throw std::invalid_argument("dY and dX must be rank-4 (NCHW).");
        }

        Tensor dY = make_tensor_nd(dy_ptr, dy_shape, DType::F32);
        Tensor dX = make_tensor_nd(dx_ptr, dx_shape, DType::F32);

        AvgPool2DWorkspaceBwd ws_local{};
        const AvgPool2DWorkspaceBwd* ws = nullptr;
        if (!ws_scratch_ptr_or_none.is_none()) {
          ws_local.scratch = reinterpret_cast<float*>(ws_scratch_ptr_or_none.cast<uintptr_t>());
          ws = &ws_local;
        }

        StreamHandle s = reinterpret_cast<StreamHandle>(stream_ptr);
        auto st = AvgPool2DBackwardCudaLaunch(dY, dX, attrs, s, /*ws_bwd=*/ws);
        throw_if_bad(st, "AvgPool2DBackwardCudaLaunch");
      },
    py::arg("dy_ptr"), py::arg("dy_shape"),
    py::arg("dx_ptr"), py::arg("dx_shape"),
    py::arg("ws_scratch_ptr") = py::none(),
    py::arg("attrs"),
    py::arg("stream") = static_cast<uintptr_t>(0)
  );
}
