// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "optimizers.cpp"  // 해당 부분을 실제 파일 이름으로 바꿔주세요.

namespace py = pybind11;

PYBIND11_MODULE(optimizers, m) {
    py::class_<SGD>(m, "SGD")
        .def(py::init<double>())
        .def("update", &SGD::update, "Update weights using SGD", py::arg("weights"), py::arg("gradients"));

    py::class_<Adam>(m, "Adam")
        .def(py::init<double, double, double, double>())
        .def("update", &Adam::update, "Update weights using Adam", py::arg("weights"), py::arg("gradients"));
}
