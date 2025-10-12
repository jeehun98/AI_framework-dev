// src/bindings/concat_pybind.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif
#include "backends/cuda/ops/concat/api.hpp"

namespace py=pybind11; using namespace ai;

static Tensor make_f32(uintptr_t p, const std::vector<int64_t>& sh){
  Tensor t; t.data=(void*)p; t.device=Device::CUDA; t.device_index=0;
  t.desc={DType::F32, Layout::RowMajor, sh, {}};
  t.desc.stride.resize(sh.size()); int64_t s=1;
  for(int i=(int)sh.size()-1;i>=0;--i){ t.desc.stride[i]=s; s*=sh[i]; }
  return t;
}
PYBIND11_MODULE(_ops_concat, m){
  m.attr("__package__")="graph_executor_v2.ops";
  py::class_<ConcatAttrs>(m,"ConcatAttrs").def(py::init<>()).def_readwrite("axis",&ConcatAttrs::axis);

  m.def("forward",[](std::vector<uintptr_t> in_ptrs,
                     std::vector<std::vector<int64_t>> in_shapes,
                     uintptr_t out_ptr, std::vector<int64_t> out_shape,
                     ConcatAttrs attrs, uintptr_t stream){
    if (in_ptrs.size()!=in_shapes.size()) throw std::invalid_argument("inputs mismatch");
    std::vector<Tensor> ins(in_ptrs.size());
    for(size_t i=0;i<in_ptrs.size();++i) ins[i]=make_f32(in_ptrs[i], in_shapes[i]);
    Tensor out = make_f32(out_ptr, out_shape);
    auto st = ConcatCudaLaunch(ins.data(), (int)ins.size(), out, attrs, (StreamHandle)stream);
    if (st!=Status::Ok) throw std::runtime_error("concat forward failed");
  }, py::arg("in_ptrs"), py::arg("in_shapes"),
     py::arg("out_ptr"), py::arg("out_shape"),
     py::arg("attrs")=ConcatAttrs{}, py::arg("stream")=(uintptr_t)0);
}
