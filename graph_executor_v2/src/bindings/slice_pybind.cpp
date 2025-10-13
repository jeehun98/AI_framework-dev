#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
#endif

#include "backends/cuda/ops/slice/api.hpp"

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
  int64_t s = 1;
  for (int i = (int)sh.size() - 1; i >= 0; --i) {
    t.desc.stride[i] = s;
    s *= sh[i];
  }
  return t;
}

static void throw_if_bad(Status st, const char* where){
  if (st != Status::Ok){
    throw std::runtime_error(std::string("[_ops_slice::") + where + "] failed");
  }
}

PYBIND11_MODULE(_ops_slice, m) {
  m.attr("__package__") = "graph_executor_v2.ops";
  m.doc() = "Slice op binding (I32 attrs; rank<=4)";

  py::class_<SliceAttrs>(m, "SliceAttrs")
    .def(py::init<>())
    .def_readwrite("rank", &SliceAttrs::rank)
    .def_property(
      "starts",
      [](const SliceAttrs& a){ return std::vector<int>{a.starts[0],a.starts[1],a.starts[2],a.starts[3]}; },
      [](SliceAttrs& a, const std::vector<int>& v){
        int n = (int)std::min<size_t>(4, v.size());
        for (int i=0;i<n;++i) a.starts[i] = v[i];
        for (int i=n;i<4;++i) a.starts[i] = 0;
      }
    )
    .def_property(
      "sizes",
      [](const SliceAttrs& a){ return std::vector<int>{a.sizes[0],a.sizes[1],a.sizes[2],a.sizes[3]}; },
      [](SliceAttrs& a, const std::vector<int>& v){
        int n = (int)std::min<size_t>(4, v.size());
        for (int i=0;i<n;++i) a.sizes[i] = v[i];
        for (int i=n;i<4;++i) a.sizes[i] = 1;
      }
    );

  // forward
  m.def("forward",
    [](uintptr_t x_ptr, const std::vector<int64_t>& x_shape,
       uintptr_t y_ptr, const std::vector<int64_t>& y_shape,
       const SliceAttrs& attrs, uintptr_t stream_ptr){
      Tensor X = make_f32(x_ptr, x_shape);
      Tensor Y = make_f32(y_ptr, y_shape);
      auto st = SliceCudaLaunch(X, Y, attrs, reinterpret_cast<ai::StreamHandle>(stream_ptr));
      throw_if_bad(st, "forward");
    },
    py::arg("x_ptr"), py::arg("x_shape"),
    py::arg("y_ptr"), py::arg("y_shape"),
    py::arg("attrs") = SliceAttrs{},
    py::arg("stream") = (uintptr_t)0
  );

  // backward (gX += scatter(gY))
  m.def("backward",
    [](uintptr_t gy_ptr, const std::vector<int64_t>& gy_shape,
       uintptr_t gx_ptr, const std::vector<int64_t>& gx_shape,
       const SliceAttrs& attrs, uintptr_t stream_ptr){
      Tensor gY = make_f32(gy_ptr, gy_shape);
      Tensor gX = make_f32(gx_ptr, gx_shape);
      auto st = SliceCudaBackwardLaunch(gY, gX, attrs, reinterpret_cast<ai::StreamHandle>(stream_ptr));
      throw_if_bad(st, "backward");
    },
    py::arg("gy_ptr"), py::arg("gy_shape"),
    py::arg("gx_ptr"), py::arg("gx_shape"),
    py::arg("attrs") = SliceAttrs{},
    py::arg("stream") = (uintptr_t)0
  );
}
