// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "activations.cpp"  // 해당 부분을 실제 파일 이름으로 바꿔주세요.

namespace py = pybind11;

PYBIND11_MODULE(activations, m) {
    m.def("relu", &relu, "ReLU activation function");
    m.def("sigmoid", &sigmoid, "Sigmoid activation function");
    m.def("tanh_activation", &tanh_activation, "Tanh activation function");
    m.def("leaky_relu", &leaky_relu, py::arg("inputs"), py::arg("alpha") = 0.01, "Leaky ReLU activation function");
    m.def("softmax", &softmax, "Softmax function");
}