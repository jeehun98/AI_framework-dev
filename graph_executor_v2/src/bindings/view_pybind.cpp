#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
#endif

#include "backends/cuda/ops/view/api.hpp"

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
  if (st != Status::Ok) throw std::runtime_error(std::string("[_ops_view::")+where+"] failed");
}

PYBIND11_MODULE(_ops_view, m){
  m.attr("__package__") = "graph_executor_v2.ops";
  m.doc() = "View (alias) op with backward (rank<=4, int32 attrs)";

  py::class_<ViewAttrs>(m, "ViewAttrs")
    .def(py::init<>())
    .def_readwrite("rank", &ViewAttrs::rank)
    .def_property(
      "shape",
      [](const ViewAttrs& a){
        return std::vector<int>{a.shape[0],a.shape[1],a.shape[2],a.shape[3]};
      },
      [](ViewAttrs& a, const std::vector<int>& v){
        int n = (int)std::min<size_t>(4, v.size());
        for (int i=0;i<n;++i) a.shape[i] = v[i];
        for (int i=n;i<4;++i) a.shape[i] = 1;
      }
    );

  m.def("forward",
    [](uintptr_t x_ptr, const std::vector<int64_t>& x_shape,
       uintptr_t y_ptr, const std::vector<int64_t>& y_shape,
       const ViewAttrs& attrs, uintptr_t stream_ptr){
      Tensor X = make_f32(x_ptr, x_shape);
      Tensor Y = make_f32(y_ptr, y_shape);
      auto st = ViewCudaLaunch(X, Y, attrs, reinterpret_cast<ai::StreamHandle>(stream_ptr));
      throw_if_bad(st, "forward");
    },
    py::arg("x_ptr"), py::arg("x_shape"),
    py::arg("y_ptr"), py::arg("y_shape"),
    py::arg("attrs") = ViewAttrs{},
    py::arg("stream") = (uintptr_t)0
  );

  m.def("backward",
    [](uintptr_t gy_ptr, const std::vector<int64_t>& gy_shape,
       uintptr_t gx_ptr, const std::vector<int64_t>& gx_shape,
       const ViewAttrs& attrs, uintptr_t stream_ptr){
      Tensor gY = make_f32(gy_ptr, gy_shape);
      Tensor gX = make_f32(gx_ptr, gx_shape);
      auto st = ViewCudaBackwardLaunch(gY, gX, attrs, reinterpret_cast<ai::StreamHandle>(stream_ptr));
      throw_if_bad(st, "backward");
    },
    py::arg("gy_ptr"), py::arg("gy_shape"),
    py::arg("gx_ptr"), py::arg("gx_shape"),
    py::arg("attrs") = ViewAttrs{},
    py::arg("stream") = (uintptr_t)0
  );
}
