#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "backends/cuda/ops/epilogue/api/epilogue.h"
#include "backends/cuda/ops/epilogue/api/dtype.h"

namespace py = pybind11;
using namespace epi;

namespace {

uint64_t ptr_from_capsule(py::capsule cap) {
  return reinterpret_cast<uint64_t>(cap.get_pointer());
}

void* ptr_from_obj(const py::object& o) {
  if (o.is_none()) return nullptr;

  // 1) 정수 주소
  if (py::isinstance<py::int_>(o)) {
    auto v = o.cast<uint64_t>();
    return reinterpret_cast<void*>(v);
  }
  // 2) capsule (예: 외부에서 cuda ptr을 capsule로 넘길 때)
  if (py::isinstance<py::capsule>(o)) {
    auto cap = py::cast<py::capsule>(o);
    return cap.get_pointer();
  }

  throw std::runtime_error("pointer must be int (device addr) or capsule or None");
}

} // anon

PYBIND11_MODULE(_ops_epilogue, m) {
  py::enum_<DType>(m, "DType")
    .value("F32", DType::F32)
    .value("F16", DType::F16);

  py::enum_<ActKind>(m, "ActKind")
    .value("None_", ActKind::None)
    .value("ReLU",  ActKind::ReLU)
    .value("GELU",  ActKind::GELU);

  py::enum_<BiasKind>(m, "BiasKind")
    .value("None_", BiasKind::None)
    .value("PerN",  BiasKind::PerN);

  py::class_<Attrs>(m, "Attrs")
    .def(py::init<>())
    .def_readwrite("act",      &Attrs::act)
    .def_readwrite("bias",     &Attrs::bias)
    .def_readwrite("alpha",    &Attrs::alpha)
    .def_readwrite("beta",     &Attrs::beta)
    .def_readwrite("dropout",  &Attrs::dropout)
    .def_readwrite("p_drop",   &Attrs::p_drop);

  py::class_<Plan>(m, "Plan")
    .def(py::init<>())
    .def_readwrite("attrs",     &Plan::attrs)
    .def_readwrite("sm_target", &Plan::sm_target);

  py::class_<Tensors>(m, "Tensors")
    .def(py::init<>())
    .def_readwrite("M", &Tensors::M)
    .def_readwrite("N", &Tensors::N)
    .def_readwrite("ld_x", &Tensors::ld_x)
    .def_readwrite("ld_y", &Tensors::ld_y)
    .def_readwrite("rng_seed", &Tensors::rng_seed)
    .def_readwrite("rng_offset", &Tensors::rng_offset)
    .def_property("x",
      [](const Tensors&){ return 0ull; },
      [](Tensors& t, const py::object& o){ t.x = ptr_from_obj(o); })
    .def_property("y",
      [](const Tensors&){ return 0ull; },
      [](Tensors& t, const py::object& o){ t.y = ptr_from_obj(o); })
    .def_property("bias",
      [](const Tensors&){ return 0ull; },
      [](Tensors& t, const py::object& o){ t.bias = ptr_from_obj(o); })
    .def_property("resid",
      [](const Tensors&){ return 0ull; },
      [](Tensors& t, const py::object& o){ t.resid = ptr_from_obj(o); });

  m.def("run", [](const Plan& plan, const Tensors& ts,
                 DType xdt, DType ydt, DType bdt,
                 py::object stream_obj)
  {
    void* stream = nullptr;
    if (!stream_obj.is_none()) {
      // pytorch cudaStream? 정수 주소?
      if (py::isinstance<py::int_>(stream_obj)) {
        auto v = stream_obj.cast<uint64_t>();
        stream = reinterpret_cast<void*>(v);
      } else if (py::isinstance<py::capsule>(stream_obj)) {
        auto cap = py::cast<py::capsule>(stream_obj);
        stream = cap.get_pointer();
      } else {
        throw std::runtime_error("stream must be int (addr) or capsule or None");
      }
    }
    auto st = epi::run(plan, ts, xdt, ydt, bdt, stream);
    if (!st.ok) throw std::runtime_error(st.msg);
  },
  py::arg("plan"), py::arg("tensors"),
  py::arg("xdt"), py::arg("ydt"), py::arg("bdt"),
  py::arg("stream") = py::none());
}
