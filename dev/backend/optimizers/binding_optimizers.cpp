#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "optimizers.cpp"

namespace py = pybind11;

PYBIND11_MODULE(optimizers, m){
    py::class_<SGD>(m, "SGD")
        .def(py::init<double>())
        .def("update", &SGD::update, "Update weights using SGD", py::arg("weight"), py::arg("gradients"));

    py::class_<Adam>(m, "Adam")
        .def(py::init<double, double, double, double>())
        .def("update", &Adam::update, "Update weights using Adam", py::arg("weight"), py::arg("gradients"));
}