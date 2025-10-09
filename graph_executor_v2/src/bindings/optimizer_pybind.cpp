// src/bindings/optimizer_pybind.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

#include "backends/cuda/ops/optimizer/api.hpp"

namespace py = pybind11;
using namespace ai;

// ---------------------- helpers ----------------------
static Tensor make_tensor_1d(uintptr_t ptr_u64,
                             int64_t N,
                             DType dtype=DType::F32,
                             Device dev=Device::CUDA)
{
  Tensor t;
  t.data = reinterpret_cast<void*>(ptr_u64);
  t.device = dev;
  t.device_index = 0;
  t.desc.dtype  = dtype;
  t.desc.layout = Layout::RowMajor;
  t.desc.shape  = {N};
  t.desc.stride = {1};
  return t;
}

static void throw_if_bad(Status st, const char* where){
  if (st!=Status::Ok)
    throw std::runtime_error(std::string(where)+" failed with Status="+std::to_string((int)st));
}

static int64_t require_1d_and_getN(const std::vector<int64_t>& shape, const char* name){
  if (shape.size()!=1)
    throw std::invalid_argument(std::string(name)+" must be 1D");
  if (shape[0] <= 0)
    throw std::invalid_argument(std::string(name)+" length must be > 0");
  return shape[0];
}

PYBIND11_MODULE(_ops_optimizer, m){
  m.doc() = "Independent CUDA Optimizers (SGD/AdamW)";

  // --------- Attrs classes ----------
  py::class_<SGDAttrs>(m, "SGDAttrs")
    .def(py::init<>())
    .def_readwrite("lr",           &SGDAttrs::lr)
    .def_readwrite("momentum",     &SGDAttrs::momentum)
    .def_readwrite("dampening",    &SGDAttrs::dampening)
    .def_readwrite("nesterov",     &SGDAttrs::nesterov)
    .def_readwrite("weight_decay", &SGDAttrs::weight_decay);

  py::class_<AdamWAttrs>(m, "AdamWAttrs")
    .def(py::init<>())
    .def_readwrite("lr",               &AdamWAttrs::lr)
    .def_readwrite("beta1",            &AdamWAttrs::beta1)
    .def_readwrite("beta2",            &AdamWAttrs::beta2)
    .def_readwrite("eps",              &AdamWAttrs::eps)
    .def_readwrite("weight_decay",     &AdamWAttrs::weight_decay)
    .def_readwrite("bias_correction",  &AdamWAttrs::bias_correction)
    .def_readwrite("step",             &AdamWAttrs::step);

  // ===================== SGD =====================
  // In-place: P, (optional) V
  m.def("sgd_update",
    [](uintptr_t p_ptr, const std::vector<int64_t>& p_shape,
       uintptr_t g_ptr, const std::vector<int64_t>& g_shape,
       py::object v_ptr_or_none, const std::vector<int64_t>& v_shape,
       const SGDAttrs& attrs, uintptr_t stream_ptr)
    {
      const int64_t Np = require_1d_and_getN(p_shape, "P");
      const int64_t Ng = require_1d_and_getN(g_shape, "G");
      if (Np != Ng) throw std::invalid_argument("P and G length must match");

      Tensor P = make_tensor_1d(p_ptr, Np, DType::F32);
      Tensor G = make_tensor_1d(g_ptr, Ng, DType::F32);

      Tensor VT{}; Tensor* V = nullptr;
      if (!v_ptr_or_none.is_none()){
        const int64_t Nv = require_1d_and_getN(v_shape, "V");
        if (Nv != Np) throw std::invalid_argument("V length must match P");
        auto vptr = v_ptr_or_none.cast<uintptr_t>();
        VT = make_tensor_1d(vptr, Nv, DType::F32);
        V = &VT;
      }

      auto st = SGDCudaUpdateLaunch(
        P, G, V, attrs, reinterpret_cast<StreamHandle>(stream_ptr)
      );
      throw_if_bad(st, "SGDCudaUpdateLaunch");
    },
    py::arg("p_ptr"), py::arg("p_shape"),
    py::arg("g_ptr"), py::arg("g_shape"),
    py::arg("v_ptr")=py::none(), py::arg("v_shape")=std::vector<int64_t>{},
    py::arg("attrs"),
    py::arg("stream")=static_cast<uintptr_t>(0)
  );

  // ===================== AdamW =====================
  // In-place: P, M, V (moments)
  m.def("adamw_update",
    [](uintptr_t p_ptr, const std::vector<int64_t>& p_shape,
       uintptr_t g_ptr, const std::vector<int64_t>& g_shape,
       uintptr_t m_ptr, const std::vector<int64_t>& m_shape,
       uintptr_t v_ptr, const std::vector<int64_t>& v_shape,
       const AdamWAttrs& attrs, uintptr_t stream_ptr)
    {
      const int64_t Np = require_1d_and_getN(p_shape, "P");
      const int64_t Ng = require_1d_and_getN(g_shape, "G");
      const int64_t Nm = require_1d_and_getN(m_shape, "M");
      const int64_t Nv = require_1d_and_getN(v_shape, "V");
      if (Np!=Ng || Np!=Nm || Np!=Nv)
        throw std::invalid_argument("Lengths of P,G,M,V must all match");

      Tensor P = make_tensor_1d(p_ptr, Np, DType::F32);
      Tensor G = make_tensor_1d(g_ptr, Ng, DType::F32);
      Tensor M = make_tensor_1d(m_ptr, Nm, DType::F32);
      Tensor V = make_tensor_1d(v_ptr, Nv, DType::F32);

      auto st = AdamWCudaUpdateLaunch(
        P, G, M, V, attrs, reinterpret_cast<StreamHandle>(stream_ptr)
      );
      throw_if_bad(st, "AdamWCudaUpdateLaunch");
    },
    py::arg("p_ptr"), py::arg("p_shape"),
    py::arg("g_ptr"), py::arg("g_shape"),
    py::arg("m_ptr"), py::arg("m_shape"),
    py::arg("v_ptr"), py::arg("v_shape"),
    py::arg("attrs"),
    py::arg("stream")=static_cast<uintptr_t>(0)
  );
}
