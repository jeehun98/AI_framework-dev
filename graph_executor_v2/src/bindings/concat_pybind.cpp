#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
#endif

#include "backends/cuda/ops/concat/api.hpp"

namespace py = pybind11;
using namespace ai;

static Tensor make_f32(uintptr_t p, const std::vector<int64_t>& sh){
  Tensor t;
  t.data = reinterpret_cast<void*>(p);
  t.device = Device::CUDA;
  t.device_index = 0;
  t.desc.dtype = DType::F32;
  t.desc.layout = Layout::RowMajor;
  t.desc.shape = sh;
  t.desc.stride.resize(sh.size());
  int64_t s=1;
  for (int i=(int)sh.size()-1;i>=0;--i){ t.desc.stride[i]=s; s*=sh[i]; }
  return t;
}

static void throw_if_bad(Status st, const char* where){
  if (st != Status::Ok) throw std::runtime_error(std::string("[_ops_concat::")+where+"] failed");
}

PYBIND11_MODULE(_ops_concat, m){
  m.attr("__package__") = "graph_executor_v2.ops";
  m.doc() = "Concat op (rank<=4, int32 attrs)";

  py::class_<ConcatAttrs>(m, "ConcatAttrs")
    .def(py::init<>())
    .def_readwrite("rank", &ConcatAttrs::rank)
    .def_readwrite("axis", &ConcatAttrs::axis);

  // forward: (x_ptrs, x_shapes list) -> y
  m.def("forward",
    [](const std::vector<uintptr_t>& x_ptrs,
       const std::vector<std::vector<int64_t>>& x_shapes,
       uintptr_t y_ptr, const std::vector<int64_t>& y_shape,
       ConcatAttrs attrs, uintptr_t stream_ptr){
      if (x_ptrs.size()!=x_shapes.size()) throw std::invalid_argument("x_ptrs/x_shapes size mismatch");
      std::vector<Tensor> Xs(x_ptrs.size());
      for (size_t i=0;i<x_ptrs.size();++i) Xs[i] = make_f32(x_ptrs[i], x_shapes[i]);
      Tensor Y = make_f32(y_ptr, y_shape);
      auto st = ConcatCudaLaunch(Xs.data(), (int)Xs.size(), Y, attrs,
                                 reinterpret_cast<ai::StreamHandle>(stream_ptr));
      throw_if_bad(st, "forward");
    },
    py::arg("x_ptrs"), py::arg("x_shapes"),
    py::arg("y_ptr"), py::arg("y_shape"),
    py::arg("attrs") = ConcatAttrs{},
    py::arg("stream") = (uintptr_t)0
  );

  // backward: (gy) -> list of gx (in-place add)
  m.def("backward",
    [](uintptr_t gy_ptr, const std::vector<int64_t>& gy_shape,
       const std::vector<uintptr_t>& gx_ptrs,
       const std::vector<std::vector<int64_t>>& gx_shapes,
       ConcatAttrs attrs, uintptr_t stream_ptr){
      if (gx_ptrs.size()!=gx_shapes.size()) throw std::invalid_argument("gx_ptrs/gx_shapes size mismatch");
      Tensor gY = make_f32(gy_ptr, gy_shape);
      std::vector<Tensor> gXs(gx_ptrs.size());
      for (size_t i=0;i<gx_ptrs.size();++i) gXs[i] = make_f32(gx_ptrs[i], gx_shapes[i]);
      auto st = ConcatCudaBackwardLaunch(gY, gXs.data(), (int)gXs.size(), attrs,
                                         reinterpret_cast<ai::StreamHandle>(stream_ptr));
      throw_if_bad(st, "backward");
    },
    py::arg("gy_ptr"), py::arg("gy_shape"),
    py::arg("gx_ptrs"), py::arg("gx_shapes"),
    py::arg("attrs") = ConcatAttrs{},
    py::arg("stream") = (uintptr_t)0
  );
}
