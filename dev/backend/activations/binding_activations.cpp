#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <memory>
#include "activations.cpp"  // 활성화 함수 정의 포함

namespace py = pybind11;

PYBIND11_MODULE(activations, m) {
    m.doc() = "Activation functions with computation graph support";

    m.def("relu", &relu, py::arg("inputs"), "ReLU activation function");
    m.def("sigmoid", &sigmoid, py::arg("inputs"), "Sigmoid activation function");
    m.def("tanh", &tanh_activation, py::arg("inputs"), "Tanh activation function");
    m.def("leaky_relu", &leaky_relu, py::arg("inputs"), py::arg("alpha") = 0.01, "Leaky ReLU activation function");
    m.def("softmax", &softmax, py::arg("inputs"), "Softmax activation function");
}
