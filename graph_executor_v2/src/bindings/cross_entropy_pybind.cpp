// src/bindings/cross_entropy_pybind.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>


#include "backends/cuda/ops/_common/shim/ai_shim.hpp"

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
  m.doc() = "CUDA Cross-Entropy: logits/probs fwd/bwd + fused softmax+CE backward (int32 labels)";

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

  // === SCEFuseAttrs 노출 ===
  py::class_<SCEFuseAttrs>(m, "SCEFuseAttrs")
    .def(py::init<>())
    .def_readwrite("stable", &SCEFuseAttrs::stable)
    .def_readwrite("reduction", &SCEFuseAttrs::reduction);

  py::enum_<SCEFuseAttrs::Reduction>(m, "SCEFuseReduction")
    .value("None", SCEFuseAttrs::Reduction::None)
    .value("Mean", SCEFuseAttrs::Reduction::Mean)
    .value("Sum",  SCEFuseAttrs::Reduction::Sum)
    .export_values();
  
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


  // === fused forward+backward (from logits) ===
  m.def("fused_forward_backward",
    [](uintptr_t logits_ptr, const std::vector<int64_t>& logits_shape,
       uintptr_t labels_ptr, const std::vector<int64_t>& labels_shape,
       uintptr_t dlogits_ptr, const std::vector<int64_t>& dlogits_shape,
       py::object loss_ptr_or_none, const std::vector<int64_t>& loss_shape,
       const SCEFuseAttrs& attrs, uintptr_t stream_ptr)
    {
      if (logits_shape.size()!=2) throw std::invalid_argument("logits must be [M,C]");
      if (dlogits_shape != logits_shape) throw std::invalid_argument("dlogits shape must match logits");
      if (labels_shape.size()!=1 || labels_shape[0] != logits_shape[0])
        throw std::invalid_argument("labels must be [M] and match logits M");

      Tensor L  = make_tensor_nd(logits_ptr,  logits_shape,  DType::F32);
      Tensor T  = make_tensor_nd(labels_ptr,  labels_shape,  DType::I32);
      Tensor dL = make_tensor_nd(dlogits_ptr, dlogits_shape, DType::F32);

      Tensor lossT{}; Tensor* lossPtr=nullptr;
      if (!loss_ptr_or_none.is_none()){
        lossT = make_tensor_nd(loss_ptr_or_none.cast<uintptr_t>(), loss_shape, DType::F32);
        lossPtr = &lossT;
      }

      auto st = SoftmaxCEFusedForwardBackwardCudaLaunch(
        L, T, dL, lossPtr, attrs, reinterpret_cast<StreamHandle>(stream_ptr)
      );
      throw_if_bad(st, "SoftmaxCEFusedForwardBackwardCudaLaunch");
    },
    py::arg("logits_ptr"), py::arg("logits_shape"),
    py::arg("labels_ptr"), py::arg("labels_shape"),
    py::arg("dlogits_ptr"), py::arg("dlogits_shape"),
    py::arg("loss_ptr")=py::none(), py::arg("loss_shape")=std::vector<int64_t>{},
    py::arg("attrs"),
    py::arg("stream")=static_cast<uintptr_t>(0)
  );
}
