// pybind/epilogue_bind.cpp
#include <pybind11/pybind11.h>
#include "api/epilogue.h"
namespace py = pybind11;
PYBIND11_MODULE(_ops_epilogue, m) {
  py::enum_<DType>(m,"DType").value("F32",DType::F32);
  py::class_<epi::Attrs>(m,"Attrs")
     .def(py::init<>())
     .def_readwrite("alpha",&epi::Attrs::alpha) /* ... 기타 생략 ... */;
  py::class_<epi::Tensors>(m,"Tensors")
     .def(py::init<>())
     .def_readwrite("x",&epi::Tensors::x) /* ... */;
  py::class_<epi::Plan>(m,"Plan").def(py::init<>());
  m.def("run",&epi::run, "Run epilogue");
}
