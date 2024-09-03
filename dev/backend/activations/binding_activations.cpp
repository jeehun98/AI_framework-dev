#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "activations.cpp"

namespace py = pybind11;

PYBIND11_MODULE(acativations, m){
    m.def("relu", &relu, "ReLU activation function");
    m.def("sigmoid", &sigmoid, "Sigmoid activation function");
    m.def("tanh_activation", &tanh_activation, "Tanh activation function");
    m.def("leaky_relu", &leaky_relu, py::arg("x"), py::arg("alpha") = 0.01, "Leaky ReLU activation function");
    m.def("softmax", &softmax, "Softmax activtion function");
}