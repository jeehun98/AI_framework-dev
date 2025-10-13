// src/bindings/embedding_pybind.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

#include "backends/cuda/ops/embedding/api.hpp"

namespace py = pybind11;
using namespace ai;

static Tensor make_tensor(uintptr_t ptr_u64,
                          const std::vector<int64_t>& shape,
                          DType dtype, Device dev=Device::CUDA){
  Tensor t;
  t.data = reinterpret_cast<void*>(ptr_u64);
  t.device = dev; t.device_index = 0;
  t.desc.dtype = dtype; t.desc.layout = Layout::RowMajor;
  t.desc.shape = shape;
  t.desc.stride.resize(shape.size());
  int64_t s = 1;
  for (int i=(int)shape.size()-1; i>=0; --i) {
    t.desc.stride[i] = s;
    s *= shape[i];
  }
  return t;
}

static void throw_if_bad(Status st, const char* where){
  if (st != Status::Ok) {
    throw std::runtime_error(std::string("[_ops_embedding::") + where + "] failed: " +
                             std::to_string((int)st));
  }
}

PYBIND11_MODULE(_ops_embedding, m) {
  m.attr("__package__") = "graph_executor_v2.ops";
  m.doc() = "Independent embedding CUDA ops binding";

  py::class_<EmbeddingAttrs>(m, "EmbeddingAttrs")
    .def(py::init<>())
    .def_readwrite("padding_idx",        &EmbeddingAttrs::padding_idx)
    .def_readwrite("scale_grad_by_freq", &EmbeddingAttrs::scale_grad_by_freq)
    .def_readwrite("out_scale",          &EmbeddingAttrs::out_scale);

m.def("forward",
[](uintptr_t w_ptr, const std::vector<int64_t>& w_shape,  // [V,D]
    uintptr_t i_ptr, const std::vector<int64_t>& i_shape,  // [N,L] or [L]
    uintptr_t y_ptr, const std::vector<int64_t>& y_shape,  // [N,L,D] or [L,D]
    EmbeddingAttrs attrs,
    uintptr_t stream){
    Tensor W = make_tensor(w_ptr, w_shape, DType::F32);
    // **중요**: 인덱스는 I32로 해석 (I64 미지원)
    Tensor I = make_tensor(i_ptr, i_shape, DType::I32);
    Tensor Y = make_tensor(y_ptr, y_shape, DType::F32);
    auto st = EmbeddingCudaLaunch(W, I, Y, attrs, reinterpret_cast<StreamHandle>(stream));
    throw_if_bad(st, "forward");
},
    
    py::arg("w_ptr"), py::arg("w_shape"),
    py::arg("i_ptr"), py::arg("i_shape"),
    py::arg("y_ptr"), py::arg("y_shape"),
    py::arg("attrs") = EmbeddingAttrs{},
    py::arg("stream") = (uintptr_t)0
  );

m.def("backward",
[](uintptr_t i_ptr, const std::vector<int64_t>& i_shape,
    uintptr_t gy_ptr, const std::vector<int64_t>& gy_shape,
    py::object dW_ptr_obj, const std::vector<int64_t>& w_shape,
    EmbeddingAttrs attrs, uintptr_t stream){
    Tensor I  = make_tensor(i_ptr,  i_shape,  DType::I32);  // I32 고정
    Tensor dY = make_tensor(gy_ptr, gy_shape, DType::F32);
    Tensor* dW = nullptr; Tensor dW_T;
    if (!dW_ptr_obj.is_none()) {
    auto p = dW_ptr_obj.cast<uintptr_t>();
    dW_T = make_tensor(p, w_shape, DType::F32);
    dW = &dW_T;
    }
    auto st = EmbeddingCudaBackwardLaunch(I, dY, dW, attrs, reinterpret_cast<StreamHandle>(stream));
    throw_if_bad(st, "backward");
},
    py::arg("i_ptr"),  py::arg("i_shape"),
    py::arg("gy_ptr"), py::arg("gy_shape"),
    py::arg("dW_ptr") = py::none(), py::arg("w_shape") = std::vector<int64_t>{},
    py::arg("attrs") = EmbeddingAttrs{},
    py::arg("stream") = (uintptr_t)0
  );
}
