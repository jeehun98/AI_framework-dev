// src/bindings/slice_pybind.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif
#include "backends/cuda/ops/slice/api.hpp"
namespace py=pybind11; using namespace ai;

static Tensor make(uintptr_t p,const std::vector<int64_t>&sh){
  Tensor t; t.data=(void*)p; t.device=Device::CUDA; t.device_index=0;
  t.desc={DType::F32, Layout::RowMajor, sh, {}}; t.desc.stride.resize(sh.size());
  int64_t s=1; for(int i=(int)sh.size()-1;i>=0;--i){t.desc.stride[i]=s; s*=sh[i];}
  return t;
}
PYBIND11_MODULE(_ops_slice, m){
  m.attr("__package__")="graph_executor_v2.ops";
  py::class_<SliceAttrs>(m,"SliceAttrs")
    .def(py::init<>())
    .def_readwrite("starts",&SliceAttrs::starts)
    .def_readwrite("sizes",&SliceAttrs::sizes)
    .def_readwrite("rank",&SliceAttrs::rank);

  m.def("forward",[](uintptr_t x_ptr, std::vector<int64_t> x_shape,
                     uintptr_t y_ptr, std::vector<int64_t> y_shape,
                     SliceAttrs attrs, uintptr_t stream){
    Tensor X=make(x_ptr,x_shape), Y=make(y_ptr,y_shape);
    auto st = SliceCudaLaunch(X,Y,attrs,(StreamHandle)stream);
    if (st!=Status::Ok) throw std::runtime_error("slice forward failed");
  }, py::arg("x_ptr"), py::arg("x_shape"),
     py::arg("y_ptr"), py::arg("y_shape"),
     py::arg("attrs")=SliceAttrs{}, py::arg("stream")=(uintptr_t)0);
}
