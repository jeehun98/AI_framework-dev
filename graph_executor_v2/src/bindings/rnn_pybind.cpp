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
  int64_t st = 1;
  for (int i = (int)shape.size()-1; i >= 0; --i) { s[i] = st; st *= shape[i]; }
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
  if (st != Status::Ok) {
    throw py::value_error(std::string("[_ops_rnn] ") + where +
                          " failed (ai::Status=" + std::to_string((int)st) + ")");
  }
}

PYBIND11_MODULE(_ops_rnn, m){
  m.doc() = "Vanilla RNN (tanh) forward/backward (CUDA, capture-safe)";

  // ---- Minimal exposure for conversions / debug ----
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

  // ---- Workspace structs (opaque in Python; build via factory helpers) ----
  py::class_<RNNWorkspaceFwd>(m, "RNNWorkspaceFwd")
    .def(py::init<>());

  py::class_<RNNWorkspaceBwd>(m, "RNNWorkspaceBwd")
    .def(py::init<>());

  // Factories for workspaces: pass device pointers as uintptr_t
  m.def("make_ws_fwd",
    [](uintptr_t prez_ptr, uintptr_t tmp_h_ptr, uintptr_t tmp_z_ptr){
      RNNWorkspaceFwd ws{};
      ws.PreZ_all = reinterpret_cast<float*>(prez_ptr);
      ws.TMP_H    = reinterpret_cast<float*>(tmp_h_ptr);
      ws.TMP_Z    = reinterpret_cast<float*>(tmp_z_ptr);
      return ws;
    },
    py::arg("PreZ_all_ptr"), py::arg("TMP_H_ptr"), py::arg("TMP_Z_ptr"),
    py::return_value_policy::move
  );

  m.def("make_ws_bwd",
    [](uintptr_t dHsum_ptr, uintptr_t dh_next_ptr, uintptr_t dZ_all_ptr, uintptr_t Hprev_all_ptr){
      RNNWorkspaceBwd ws{};
      ws.dHsum     = reinterpret_cast<float*>(dHsum_ptr);
      ws.dh_next   = reinterpret_cast<float*>(dh_next_ptr);
      ws.dZ_all    = reinterpret_cast<float*>(dZ_all_ptr);
      ws.Hprev_all = reinterpret_cast<float*>(Hprev_all_ptr);
      return ws;
    },
    py::arg("dHsum_ptr"), py::arg("dh_next_ptr"), py::arg("dZ_all_ptr"), py::arg("Hprev_all_ptr"),
    py::return_value_policy::move
  );

  // ---- Tensor factories (device, f32, row-major contiguous) ----
  m.def("make_tensor_1d", [](uintptr_t ptr_u64, const std::vector<int64_t>& shape)->Tensor{
      if (shape.size()!=1) throw std::invalid_argument("make_tensor_1d expects [N]");
      return make_tensor(ptr_u64, shape);
    }, py::arg("ptr_u64"), py::arg("shape"), py::return_value_policy::move);

  m.def("make_tensor_2d", [](uintptr_t ptr_u64, const std::vector<int64_t>& shape)->Tensor{
      if (shape.size()!=2) throw std::invalid_argument("make_tensor_2d expects [M,N]");
      return make_tensor(ptr_u64, shape);
    }, py::arg("ptr_u64"), py::arg("shape"), py::return_value_policy::move);

  m.def("make_tensor_3d", [](uintptr_t ptr_u64, const std::vector<int64_t>& shape)->Tensor{
      if (shape.size()!=3) throw std::invalid_argument("make_tensor_3d expects [T,B,H] (or [T,B,I])");
      return make_tensor(ptr_u64, shape);
    }, py::arg("ptr_u64"), py::arg("shape"), py::return_value_policy::move);

  // ================= Forward =================
  // rnn_forward(..., attrs, stream_ptr, ws_fwd=None)
  m.def("rnn_forward",
    [](const Tensor& X, const Tensor& h0, const Tensor& Wx, const Tensor& Wh,
       py::object b_or_none, Tensor& Hout, py::object Zbuf_or_none,
       const RNNAttrs& attrs, uintptr_t stream_ptr,
       py::object ws_fwd_or_none)->void
    {
      const Tensor* bptr = nullptr;
      const Tensor* zptr = nullptr;
      Tensor btmp, ztmp;

      if (!b_or_none.is_none()) {
        btmp = b_or_none.cast<Tensor>();  // expect _ops_rnn.make_tensor_1d(...)
        bptr = &btmp;
      }
      if (!Zbuf_or_none.is_none()) {
        ztmp = Zbuf_or_none.cast<Tensor>(); // expect _ops_rnn.make_tensor_2d(...)
        zptr = &ztmp;
      }

      const RNNWorkspaceFwd* ws_fwd = nullptr;
      RNNWorkspaceFwd ws_fwd_val{};
      if (!ws_fwd_or_none.is_none()) {
        ws_fwd_val = ws_fwd_or_none.cast<RNNWorkspaceFwd>();
        ws_fwd = &ws_fwd_val;
      }

      StreamHandle s = reinterpret_cast<StreamHandle>(stream_ptr);

      auto st = RNNCudaLaunch(
          X, h0, Wx, Wh,
          bptr,
          Hout,
          const_cast<Tensor*>(zptr),
          attrs,
          s,
          ws_fwd
      );
      throw_if_not_ok(st, "RNNCudaLaunch");
    },
    py::arg("X"), py::arg("h0"), py::arg("Wx"), py::arg("Wh"),
    py::arg("b") = py::none(),
    py::arg("Hout"),
    py::arg("Zbuf") = py::none(),
    py::arg("attrs"),
    py::arg("stream") = static_cast<uintptr_t>(0),
    py::arg("ws_fwd") = py::none()
  );

  // ================= Backward =================
  // rnn_backward(..., attrs, stream_ptr, ws_bwd=None)
  m.def("rnn_backward",
    [](const Tensor& X, const Tensor& Hout, py::object Zbuf_or_none, const Tensor& h0,
       const Tensor& Wx, const Tensor& Wh, const Tensor& dHout,
       Tensor& dX, Tensor& dh0, Tensor& dWx, Tensor& dWh, Tensor& dB,
       const RNNAttrs& attrs, uintptr_t stream_ptr,
       py::object ws_bwd_or_none)->void
    {
      const Tensor* zptr = nullptr;
      Tensor ztmp;
      if (!Zbuf_or_none.is_none()) {
        ztmp = Zbuf_or_none.cast<Tensor>();
        zptr = &ztmp;
      }

      const RNNWorkspaceBwd* ws_bwd = nullptr;
      RNNWorkspaceBwd ws_bwd_val{};
      if (!ws_bwd_or_none.is_none()) {
        ws_bwd_val = ws_bwd_or_none.cast<RNNWorkspaceBwd>();
        ws_bwd = &ws_bwd_val;
      }

      StreamHandle s = reinterpret_cast<StreamHandle>(stream_ptr);

      auto st = RNNCudaBackwardLaunch(
        X, Hout, zptr, h0, Wx, Wh, dHout,
        &dX, &dh0, &dWx, &dWh, &dB,
        attrs, s, ws_bwd
      );
      throw_if_not_ok(st, "RNNCudaBackwardLaunch");
    },
    py::arg("X"), py::arg("Hout"), py::arg("Zbuf") = py::none(),
    py::arg("h0"), py::arg("Wx"), py::arg("Wh"), py::arg("dHout"),
    py::arg("dX"), py::arg("dh0"), py::arg("dWx"), py::arg("dWh"), py::arg("dB"),
    py::arg("attrs"),
    py::arg("stream") = static_cast<uintptr_t>(0),
    py::arg("ws_bwd") = py::none()
  );
}
