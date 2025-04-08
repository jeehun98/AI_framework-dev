#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <memory>
#include "activations.cpp"  // 활성화 함수 정의 포함

namespace py = pybind11;

PYBIND11_MODULE(activations, m) {
    m.doc() = "Activation functions with computation graph support";

    m.def("relu", &relu, py::arg("inputs"), py::arg("node_list") = std::vector<std::shared_ptr<Node>>(), "ReLU activation function with optional node_list");
    m.def("sigmoid", &sigmoid, py::arg("inputs"), py::arg("node_list") = std::vector<std::shared_ptr<Node>>(), "Sigmoid activation function with optional node_list");
    m.def("tanh", &tanh_activation, py::arg("inputs"), py::arg("node_list") = std::vector<std::shared_ptr<Node>>(), "Tanh activation function with optional node_list");
    m.def("leaky_relu", &leaky_relu, py::arg("inputs"), py::arg("alpha") = 0.01, py::arg("node_list") = std::vector<std::shared_ptr<Node>>(), "Leaky ReLU activation function with optional node_list");
    m.def("softmax", &softmax, py::arg("inputs"), py::arg("node_list") = std::vector<std::shared_ptr<Node>>(), "Softmax activation function with optional node_list");
}
