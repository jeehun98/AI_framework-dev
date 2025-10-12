// src/bindings/view_pybind.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
#endif
#include "backends/cuda/ops/view/api.hpp"
namespace py=pybind11; using namespace ai;

static Tensor make(uintptr_t p,const std::vector<int64_t>&sh){
  Tensor t; t.data=(void*)p; t.device=Device::CUDA; t.device_index=0;
  t.desc={DType::F32, Layout::RowMajor, sh, {}}; t.desc.stride.resize(sh.size());
  int64_t s=1; for(int i=(int)sh.size()-1;i>=0;--i){ t.desc.stride[i]=s; s*=sh[i]; }
  return t;
}
PYBIND11_MODULE(_ops_view, m){
  m.attr("__package__")="graph_executor_v2.ops";
  py::class_<ViewAttrs>(m,"ViewAttrs").def(py::init<>()).def_readwrite("rank",&ViewAttrs::rank).def_readwrite("shape",&ViewAttrs::shape);
  m.def("alias_check",[](uintptr_t x,int64_t /*xbytes*/, std::vector<int64_t> xshape,
                         uintptr_t y,std::vector<int64_t> yshape, ViewAttrs a){
    Tensor X=make(x,xshape), Y=make(y,yshape);
    auto st = ViewAliasCheck(X,Y,a);
    if (st!=Status::Ok) throw std::runtime_error("view alias check failed");
  }, py::arg("x_ptr"), py::arg("x_nbytes"), py::arg("x_shape"),
     py::arg("y_ptr"), py::arg("y_shape"),
     py::arg("attrs")=ViewAttrs{});
}
