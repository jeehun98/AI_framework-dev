// src/bindings/pad_pybind.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "backends/cuda/ops/_common/shim/ai_shim.hpp"

#include "backends/cuda/ops/pad/api.hpp"

namespace py = pybind11;
using namespace ai;

// ------------------------- helpers -------------------------
static Tensor make_tensor_nd_with_strides(uintptr_t ptr_u64,
                                          const std::vector<int64_t>& shape,
                                          const std::vector<int64_t>* strides_elems, // 요소 단위 strides (선택)
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
    return t;
  }

  if (strides_elems && strides_elems->size() == R) {
    // 주어진 strides 사용 (요소 단위)
    for (size_t i = 0; i < R; ++i) t.desc.stride[i] = (*strides_elems)[i];
  } else {
    // row-major contiguous (요소 단위)
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

static void check_spec_sizes(const std::vector<int64_t>& in_shape, const PadSpec& spec) {
  if (spec.before.size() != in_shape.size() || spec.after.size() != in_shape.size()) {
    throw std::invalid_argument("PadSpec.before/after size must match tensor rank");
  }
  for (size_t i=0;i<in_shape.size();++i) {
    if (spec.before[i] < 0 || spec.after[i] < 0) {
      throw std::invalid_argument("PadSpec.before/after must be non-negative");
    }
  }
}

static void check_padded_shape(const std::vector<int64_t>& in_shape,
                               const std::vector<int64_t>& out_shape,
                               const PadSpec& spec)
{
  if (in_shape.size() != out_shape.size()) {
    throw std::invalid_argument("Input/Output rank mismatch");
  }
  for (size_t i=0;i<in_shape.size();++i) {
    const int64_t expect = in_shape[i] + (int64_t)spec.before[i] + (int64_t)spec.after[i];
    if (out_shape[i] != expect) {
      throw std::invalid_argument("Y.shape must equal X.shape + before + after (dim " + std::to_string(i) + ")");
    }
  }
}

// ------------------------- module -------------------------
PYBIND11_MODULE(_ops_pad, m) {
  m.doc() = "Independent pad CUDA op binding (capture-safe, standalone shim compatible)";

  // PadSpec
  py::class_<PadSpec>(m, "PadSpec")
    .def(py::init<>())
    .def_readwrite("before", &PadSpec::before)   // list[int], per-dim front pads
    .def_readwrite("after",  &PadSpec::after)    // list[int], per-dim back pads
    .def_readwrite("value",  &PadSpec::value);   // float fill value

  // forward: Y = pad(X, spec)
  // strides 인자는 요소 단위(ELEMENTS) strides 입니다. 생략 시 contiguous 가정.
  m.def("forward",
    [](uintptr_t x_ptr, const std::vector<int64_t>& x_shape,
       uintptr_t y_ptr, const std::vector<int64_t>& y_shape,
       const PadSpec& spec,
       py::object x_strides_elems_or_none,   // None | list[int64]
       py::object y_strides_elems_or_none,   // None | list[int64]
       uintptr_t stream_ptr) {

        if (x_shape.empty() || y_shape.empty()) {
          throw std::invalid_argument("x_shape and y_shape must be non-empty");
        }
        check_spec_sizes(x_shape, spec);
        check_padded_shape(x_shape, y_shape, spec);

        std::vector<int64_t> x_strides, y_strides;
        const std::vector<int64_t>* xs = nullptr;
        const std::vector<int64_t>* ys = nullptr;

        if (!x_strides_elems_or_none.is_none()) {
          x_strides = x_strides_elems_or_none.cast<std::vector<int64_t>>();
          if (x_strides.size() != x_shape.size())
            throw std::invalid_argument("x_strides length must match x_shape rank");
          xs = &x_strides;
        }
        if (!y_strides_elems_or_none.is_none()) {
          y_strides = y_strides_elems_or_none.cast<std::vector<int64_t>>();
          if (y_strides.size() != y_shape.size())
            throw std::invalid_argument("y_strides length must match y_shape rank");
          ys = &y_strides;
        }

        Tensor X = make_tensor_nd_with_strides(x_ptr, x_shape, xs);
        Tensor Y = make_tensor_nd_with_strides(y_ptr, y_shape, ys);

        StreamHandle s = reinterpret_cast<StreamHandle>(stream_ptr);
        auto st = PadCudaLaunch(X, Y, spec, s);
        throw_if_bad(st, "PadCudaLaunch");
      },
    py::arg("x_ptr"), py::arg("x_shape"),
    py::arg("y_ptr"), py::arg("y_shape"),
    py::arg("spec"),
    py::arg("x_strides_elems") = py::none(),
    py::arg("y_strides_elems") = py::none(),
    py::arg("stream") = static_cast<uintptr_t>(0)
  );

  // backward: dX = slice(dY, spec)
  m.def("backward",
    [](uintptr_t dy_ptr, const std::vector<int64_t>& dy_shape,
       uintptr_t dx_ptr, const std::vector<int64_t>& dx_shape,
       const PadSpec& spec,
       py::object dy_strides_elems_or_none,  // None | list[int64]
       py::object dx_strides_elems_or_none,  // None | list[int64]
       uintptr_t stream_ptr) {

        if (dy_shape.empty() || dx_shape.empty()) {
          throw std::invalid_argument("dy_shape and dx_shape must be non-empty");
        }
        // spec 크기 검증 + dY 모양 = dX + pads 검증
        check_spec_sizes(dx_shape, spec);
        check_padded_shape(dx_shape, dy_shape, spec);

        std::vector<int64_t> dy_strides, dx_strides;
        const std::vector<int64_t>* ys = nullptr;
        const std::vector<int64_t>* xs = nullptr;

        if (!dy_strides_elems_or_none.is_none()) {
          dy_strides = dy_strides_elems_or_none.cast<std::vector<int64_t>>();
          if (dy_strides.size() != dy_shape.size())
            throw std::invalid_argument("dy_strides length must match dy_shape rank");
          ys = &dy_strides;
        }
        if (!dx_strides_elems_or_none.is_none()) {
          dx_strides = dx_strides_elems_or_none.cast<std::vector<int64_t>>();
          if (dx_strides.size() != dx_shape.size())
            throw std::invalid_argument("dx_strides length must match dx_shape rank");
          xs = &dx_strides;
        }

        Tensor dY = make_tensor_nd_with_strides(dy_ptr, dy_shape, ys);
        Tensor dX = make_tensor_nd_with_strides(dx_ptr, dx_shape, xs);

        StreamHandle s = reinterpret_cast<StreamHandle>(stream_ptr);
        auto st = PadBackwardCudaLaunch(dY, dX, spec, s);
        throw_if_bad(st, "PadBackwardCudaLaunch");
      },
    py::arg("dy_ptr"), py::arg("dy_shape"),
    py::arg("dx_ptr"), py::arg("dx_shape"),
    py::arg("spec"),
    py::arg("dy_strides_elems") = py::none(),
    py::arg("dx_strides_elems") = py::none(),
    py::arg("stream") = static_cast<uintptr_t>(0)
  );
}
