// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "activations.cpp"  // 해당 부분을 실제 파일 이름으로 바꿔주세요.
#include "../node/node.h"

namespace py = pybind11;

PYBIND11_MODULE(operations_matrix, m) {
    py::class_<Node, std::shared_ptr<Node>>(m, "Node")
        .def(py::init<const std::string&, double, double>())
        .def_readwrite("operation", &Node::operation)
        .def_readwrite("input", &Node::input_a)
        .def_readwrite("output", &Node::output)
        .def_readwrite("parents", &Node::parents)
        .def_readwrite("children", &Node::children);

    m.def("softmax", &softmax, "Apply Softmax using component nodes");
}


/*
PYBIND11_MODULE(activations, m) {
    m.def("relu", &relu, "ReLU activation function");
    m.def("sigmoid", &sigmoid, "Sigmoid activation function");
    m.def("tanh_activation", &tanh_activation, "Tanh activation function");
    m.def("leaky_relu", &leaky_relu, py::arg("inputs"), py::arg("alpha") = 0.01, "Leaky ReLU activation function");
    m.def("softmax", &softmax, "Softmax function");
}
*/