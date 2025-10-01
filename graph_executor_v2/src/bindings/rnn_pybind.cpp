// src/bindings/rnn_pybind.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "backends/cuda/ops/rnn/api.hpp"

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/op_schema.hpp"
#endif

namespace py = pybind11;
using namespace ::ai;

static std::vector<int64_t> rm_strides(const std::vector<int64_t>& shape){
  std::vector<int64_t> s(shape.size());
  int64_t st=1; for (int i=(int)shape.size()-1;i>=0;--i){ s[i]=st; st*=shape[i]; }
  return s;
}
static Tensor make_tensor(uintptr_t p, const std::vector<int64_t>& shape){
  Tensor t;
  t.data         = reinterpret_cast<void*>(p);
  t.device       = Device::CUDA;
  t.device_index = 0;
  t.desc.dtype   = DType::F32;
  t.desc.layout  = Layout::RowMajor;
  t.desc.shape   = shape;
  t.desc.stride  = rm_strides(shape);
  return t;
}
static void throw_if_not_ok(Status st, const char* where){
  if (st != Status::Ok) throw py::value_error(std::string(where) + " failed (ai::Status != Ok)");
}

PYBIND11_MODULE(_ops_rnn, m){
  m.doc() = "Vanilla RNN (tanh) forward/backward";

  // minimal exposure for return conversion
  py::class_<TensorDesc>(m, "TensorDesc")
    .def(py::init<>())
    .def_readwrite("shape",  &TensorDesc::shape)
    .def_readwrite("stride", &TensorDesc::stride);

  py::class_<Tensor>(m, "Tensor")
    .def(py::init<>())
    .def_readwrite("desc", &Tensor::desc);

  py::class_<RNNAttrs>(m, "RNNAttrs")
    .def(py::init<>())
    .def_readwrite("T", &RNNAttrs::T)
    .def_readwrite("B", &RNNAttrs::B)
    .def_readwrite("I", &RNNAttrs::I)
    .def_readwrite("H", &RNNAttrs::H)
    .def_readwrite("save_z", &RNNAttrs::save_z);

  // tensor factories
  m.def("make_tensor_1d", [](uintptr_t ptr_u64, const std::vector<int64_t>& shape)->Tensor{
      if (shape.size()!=1) throw std::invalid_argument("make_tensor_1d expects [N]");
      return make_tensor(ptr_u64, shape);
    }, py::arg("ptr_u64"), py::arg("shape"), py::return_value_policy::move);

  m.def("make_tensor_2d", [](uintptr_t ptr_u64, const std::vector<int64_t>& shape)->Tensor{
      if (shape.size()!=2) throw std::invalid_argument("make_tensor_2d expects [M,N]");
      return make_tensor(ptr_u64, shape);
    }, py::arg("ptr_u64"), py::arg("shape"), py::return_value_policy::move);

  m.def("make_tensor_3d", [](uintptr_t ptr_u64, const std::vector<int64_t>& shape)->Tensor{
      if (shape.size()!=3) throw std::invalid_argument("make_tensor_3d expects [T,B,H/I]");
      return make_tensor(ptr_u64, shape);
    }, py::arg("ptr_u64"), py::arg("shape"), py::return_value_policy::move);

  // forward/backward
  m.def("rnn_forward",
    [](const Tensor& X, const Tensor& h0, const Tensor& Wx, const Tensor& Wh,
       const Tensor* b, Tensor& Hout, const Tensor* Zbuf, const RNNAttrs& attrs)->void{
      auto st = RNNCudaLaunch(X,h0,Wx,Wh,b,Hout,const_cast<Tensor*>(Zbuf),attrs,nullptr);
      throw_if_not_ok(st, "RNNCudaLaunch");
    },
    py::arg("X"), py::arg("h0"), py::arg("Wx"), py::arg("Wh"),
    py::arg("b") = py::none(),
    py::arg("Hout"),
    py::arg("Zbuf") = py::none(),
    py::arg("attrs"));

  m.def("rnn_backward",
    [](const Tensor& X, const Tensor& Hout, const Tensor* Zbuf, const Tensor& h0,
       const Tensor& Wx, const Tensor& Wh, const Tensor& dHout,
       Tensor* dX, Tensor* dh0, Tensor* dWx, Tensor* dWh, Tensor* dB,
       const RNNAttrs& attrs)->void{
      auto st = RNNCudaBackwardLaunch(X,Hout,Zbuf,h0,Wx,Wh,dHout,dX,dh0,dWx,dWh,dB,attrs,nullptr);
      throw_if_not_ok(st, "RNNCudaBackwardLaunch");
    },
    py::arg("X"), py::arg("Hout"), py::arg("Zbuf") = py::none(),
    py::arg("h0"), py::arg("Wx"), py::arg("Wh"), py::arg("dHout"),
    py::arg("dX"), py::arg("dh0"), py::arg("dWx"), py::arg("dWh"), py::arg("dB"),
    py::arg("attrs"));
}
