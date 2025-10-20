#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>                 // uintptr_t
#include "../api/epilogue.h"

namespace py = pybind11;
using namespace epi;

PYBIND11_MODULE(_ops_epilogue, m) {   // <- 모듈명은 _ops_epilogue 여야 함
  py::enum_<DType>(m, "DType")
    .value("F16", DType::F16)
    .value("F32", DType::F32);

  py::enum_<ActKind>(m, "ActKind")
    .value("None", ActKind::None)
    .value("ReLU", ActKind::ReLU)
    .value("GELU", ActKind::GELU);

  py::class_<Attrs>(m, "Attrs")
    .def(py::init<>())
    .def_readwrite("act", &Attrs::act)
    .def_readwrite("dropout_p", &Attrs::dropout_p)
    .def_readwrite("seed", &Attrs::seed)
    .def_readwrite("save_mask", &Attrs::save_mask);

  py::class_<Plan>(m, "Plan")
    .def(py::init<>())
    .def_readwrite("rows", &Plan::rows)
    .def_readwrite("cols", &Plan::cols)
    .def_readwrite("ld_x", &Plan::ld_x)
    .def_readwrite("ld_y", &Plan::ld_y)
    .def_readwrite("ld_bias", &Plan::ld_bias)
    .def_readwrite("attrs", &Plan::attrs);

  // === 여기: void* 필드를 정수 주소로 get/set 가능하게 노출 ===
  py::class_<Tensors>(m, "Tensors")
    .def(py::init<>())
    .def_property(
      "x",
      [](const Tensors& t){ return reinterpret_cast<uintptr_t>(t.x); },
      [](Tensors& t, uintptr_t addr){ t.x = reinterpret_cast<void*>(addr); }
    )
    .def_property(
      "bias",
      [](const Tensors& t){ return reinterpret_cast<uintptr_t>(t.bias); },
      [](Tensors& t, uintptr_t addr){ t.bias = reinterpret_cast<void*>(addr); }
    )
    .def_property(
      "y",
      [](const Tensors& t){ return reinterpret_cast<uintptr_t>(t.y); },
      [](Tensors& t, uintptr_t addr){ t.y = reinterpret_cast<void*>(addr); }
    )
    .def_property(
      "mask_out",
      [](const Tensors& t){ return reinterpret_cast<uintptr_t>(t.mask_out); },
      [](Tensors& t, uintptr_t addr){ t.mask_out = reinterpret_cast<void*>(addr); }
    );

  m.def("run", [](const Plan& p, const Tensors& t, DType dt){
    return (int)epi::run(p, t, dt);
  }, "Run epilogue (bias+act+dropout)");
}
