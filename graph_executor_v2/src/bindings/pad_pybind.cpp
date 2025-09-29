#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

#include "backends/cuda/ops/pad/api.hpp"

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
    // scalar-like
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
PYBIND11_MODULE(_ops_pad, m) {
  m.doc() = "Independent pad CUDA op binding (standalone shim compatible)";

  // PadSpec
  py::class_<PadSpec>(m, "PadSpec")
    .def(py::init<>())
    .def_readwrite("before", &PadSpec::before)   // list[int], per-dim front pads
    .def_readwrite("after",  &PadSpec::after)    // list[int], per-dim back pads
    .def_readwrite("value",  &PadSpec::value);   // float fill value

  // forward: Y = pad(X, spec)
  m.def("forward",
    [](uintptr_t x_ptr, const std::vector<int64_t>& x_shape,
       uintptr_t y_ptr, const std::vector<int64_t>& y_shape,
       const PadSpec& spec,
       uintptr_t stream_ptr) {

        if (x_shape.empty() || y_shape.empty()) {
          throw std::invalid_argument("x_shape and y_shape must be non-empty");
        }
        if (spec.before.size() != x_shape.size() || spec.after.size() != x_shape.size()) {
          throw std::invalid_argument("PadSpec.before/after size must match rank of X");
        }

        Tensor X = make_tensor_nd(x_ptr, x_shape);
        Tensor Y = make_tensor_nd(y_ptr, y_shape);

        StreamHandle s = reinterpret_cast<StreamHandle>(stream_ptr);
        auto st = PadCudaLaunch(X, Y, spec, s);
        throw_if_bad(st, "PadCudaLaunch");
      },
    py::arg("x_ptr"), py::arg("x_shape"),
    py::arg("y_ptr"), py::arg("y_shape"),
    py::arg("spec"),
    py::arg("stream") = static_cast<uintptr_t>(0)
  );

  // backward: dX = slice(dY, spec)
  m.def("backward",
    [](uintptr_t dy_ptr, const std::vector<int64_t>& dy_shape,
       uintptr_t dx_ptr, const std::vector<int64_t>& dx_shape,
       const PadSpec& spec,
       uintptr_t stream_ptr) {

        if (dy_shape.empty() || dx_shape.empty()) {
          throw std::invalid_argument("dy_shape and dx_shape must be non-empty");
        }
        if (spec.before.size() != dx_shape.size() || spec.after.size() != dx_shape.size()) {
          throw std::invalid_argument("PadSpec.before/after size must match rank of dX/X");
        }

        Tensor dY = make_tensor_nd(dy_ptr, dy_shape);
        Tensor dX = make_tensor_nd(dx_ptr, dx_shape);

        StreamHandle s = reinterpret_cast<StreamHandle>(stream_ptr);
        auto st = PadBackwardCudaLaunch(dY, dX, spec, s);
        throw_if_bad(st, "PadBackwardCudaLaunch");
      },
    py::arg("dy_ptr"), py::arg("dy_shape"),
    py::arg("dx_ptr"), py::arg("dx_shape"),
    py::arg("spec"),
    py::arg("stream") = static_cast<uintptr_t>(0)
  );
}
