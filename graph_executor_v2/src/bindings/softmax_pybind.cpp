// python/bindings/softmax_pybind.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

#include "backends/cuda/ops/softmax/api.hpp"

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

static void check_mask_shape_accepts(const std::vector<int64_t>& mshape,
                                     const std::vector<int64_t>& xy_shape) {
  // 허용: [M,N], [1,N], [M,1], [N]
  if (mshape.size() == 2) {
    const auto M = xy_shape[0], N = xy_shape[1];
    const bool ok = ( (mshape[0]==M && mshape[1]==N)  // [M,N]
                   || (mshape[0]==1 && mshape[1]==N)  // [1,N]
                   || (mshape[0]==M && mshape[1]==1)  // [M,1]
                   );
    if (!ok) {
      throw std::invalid_argument("Mask shape must be [M,N], [1,N], or [M,1] for given X/Y.");
    }
  } else if (mshape.size() == 1) {
    const auto N = xy_shape[1];
    if (mshape[0] != N) {
      throw std::invalid_argument("1D Mask must be length-N (interpreted as [1,N]).");
    }
  } else {
    throw std::invalid_argument("Mask must be rank-1 [N] or rank-2 [M,N]/[1,N]/[M,1].");
  }
}

// ------------------------- module -------------------------
PYBIND11_MODULE(_ops_softmax, m) {
  m.doc() = "Independent CUDA softmax/logsoftmax op bindings (capture-safe)";

  // SoftmaxAttrs
  py::class_<SoftmaxAttrs>(m, "SoftmaxAttrs")
    .def(py::init<>())
    .def_readwrite("scale", &SoftmaxAttrs::scale)
    .def_readwrite("log",   &SoftmaxAttrs::log);

  // -------- Softmax/LogSoftmax Forward --------
  // X[M,N] (+ optional Mask[M,N] / [1,N] / [M,1] / [N]) -> Y[M,N]
  // mask: None or (uintptr_t, shape:list[int])
  m.def("softmax_forward",
    [](uintptr_t x_ptr, const std::vector<int64_t>& x_shape,
       uintptr_t y_ptr, const std::vector<int64_t>& y_shape,
       py::object mask_tuple_or_none,
       const SoftmaxAttrs& attrs,
       uintptr_t stream_ptr) {

        if (x_shape.size() != 2 || y_shape.size() != 2) {
          throw std::invalid_argument("X and Y must be rank-2 [M,N].");
        }
        if (x_shape != y_shape) {
          throw std::invalid_argument("X and Y must have the same shape [M,N].");
        }

        Tensor X = make_tensor_nd(x_ptr, x_shape, DType::F32);
        Tensor Y = make_tensor_nd(y_ptr, y_shape, DType::F32);

        Tensor maskT{};
        const Tensor* Mask = nullptr;
        if (!mask_tuple_or_none.is_none()) {
          auto tup = mask_tuple_or_none.cast<std::tuple<uintptr_t, std::vector<int64_t>>>();
          const uintptr_t mptr = std::get<0>(tup);
          const auto& mshape   = std::get<1>(tup);
          check_mask_shape_accepts(mshape, x_shape);
          maskT = make_tensor_nd(mptr, mshape, DType::F32);
          Mask = &maskT;
        }

        StreamHandle s = reinterpret_cast<StreamHandle>(stream_ptr);
        auto st = SoftmaxCudaLaunch(X, Mask, Y, attrs, s);
        throw_if_bad(st, "SoftmaxCudaLaunch");
      },
    py::arg("x_ptr"), py::arg("x_shape"),
    py::arg("y_ptr"), py::arg("y_shape"),
    py::arg("mask") = py::none(),
    py::arg("attrs"),
    py::arg("stream") = static_cast<uintptr_t>(0)
  );

  // -------- Softmax/LogSoftmax Backward --------
  // (Y_or_X)[M,N], dY[M,N] (+ optional Mask) -> dX[M,N]
  // 주: 현재 커널은 y_provided=True(Y 필요), Mask는 bwd에 영향 없음(무시)
  m.def("softmax_backward",
    [](uintptr_t y_or_x_ptr, const std::vector<int64_t>& xy_shape,
       uintptr_t dy_ptr,      const std::vector<int64_t>& dy_shape,
       uintptr_t dx_ptr,      const std::vector<int64_t>& dx_shape,
       py::object mask_tuple_or_none,   // 호환 목적 인자(런처/커널에서 무시됨)
       const SoftmaxAttrs& attrs,
       bool y_provided,
       uintptr_t stream_ptr) {

        if (xy_shape.size() != 2 || dy_shape.size() != 2 || dx_shape.size() != 2) {
          throw std::invalid_argument("All tensors must be rank-2 [M,N].");
        }
        if (xy_shape != dy_shape || xy_shape != dx_shape) {
          throw std::invalid_argument("Shapes of (Y_or_X, dY, dX) must match [M,N].");
        }

        // y_provided=False 경로는 현재 미지원(런처에서 Invalid 반환 예정) — 여기서도 사전 방지 가능
        if (!y_provided) {
          throw std::invalid_argument("softmax_backward: y_provided=False is not supported by current kernel. "
                                      "Pass Y (forward output) with y_provided=True.");
        }

        Tensor YX = make_tensor_nd(y_or_x_ptr, xy_shape, DType::F32);
        Tensor dY = make_tensor_nd(dy_ptr,      dy_shape, DType::F32);
        Tensor dX = make_tensor_nd(dx_ptr,      dx_shape, DType::F32);

        // mask는 bwd에서 사용하지 않지만, API 호환을 위해 튜플 형태는 받아 검증만 수행
        if (!mask_tuple_or_none.is_none()) {
          auto tup = mask_tuple_or_none.cast<std::tuple<uintptr_t, std::vector<int64_t>>>();
          const auto& mshape = std::get<1>(tup);
          check_mask_shape_accepts(mshape, xy_shape);
          // 전달하지 않음 (런처/커널에서 무시)
        }

        StreamHandle s = reinterpret_cast<StreamHandle>(stream_ptr);
        auto st = SoftmaxCudaBackwardLaunch(YX, /*Mask*/nullptr, dY, dX, attrs, /*y_provided*/true, s);
        throw_if_bad(st, "SoftmaxCudaBackwardLaunch");
      },
    py::arg("y_or_x_ptr"), py::arg("y_or_x_shape"),
    py::arg("dy_ptr"),     py::arg("dy_shape"),
    py::arg("dx_ptr"),     py::arg("dx_shape"),
    py::arg("mask") = py::none(),
    py::arg("attrs"),
    py::arg("y_provided") = true,
    py::arg("stream") = static_cast<uintptr_t>(0)
  );
}
