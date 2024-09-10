#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <memory>

#include "activations.cpp"  
#include "../node/node.h"

namespace py = pybind11;

// 각 활성화 함수의 선언
std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> relu(py::array_t<double> inputs);
std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> sigmoid(py::array_t<double> inputs);
std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> tanh_activation(py::array_t<double> inputs);
std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> leaky_relu(py::array_t<double> inputs, double alpha);
std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> softmax(py::array_t<double> inputs);

PYBIND11_MODULE(activations, m) {
    m.doc() = "Activation functions with computation graph support";

     py::class_<Node, std::shared_ptr<Node>>(m, "Node")
        .def(py::init<const std::string&, double, double, double>(),
             py::arg("operation"), py::arg("input_a"), py::arg("input_b"), py::arg("output"))
        .def(py::init<const std::string&, double, double>(),
             py::arg("operation"), py::arg("input_a"), py::arg("output"))
        .def("add_parent", &Node::add_parent)
        .def("add_child", &Node::add_child)
        .def("calculate_gradient", &Node::calculate_gradient)
        .def_readwrite("operation", &Node::operation)
        .def_readwrite("input_a", &Node::input_a)
        .def_readwrite("input_b", &Node::input_b)
        .def_readwrite("output", &Node::output)
        .def_readwrite("grad_a", &Node::grad_a)
        .def_readwrite("grad_b", &Node::grad_b)
        .def_readwrite("parents", &Node::parents)
        .def_readwrite("children", &Node::children);

    // ReLU 함수 바인딩
    m.def("relu", &relu, py::arg("inputs"), "ReLU activation function");

    // Sigmoid 함수 바인딩
    m.def("sigmoid", &sigmoid, py::arg("inputs"), "Sigmoid activation function");

    // Tanh 함수 바인딩
    m.def("tanh", &tanh_activation, py::arg("inputs"), "Tanh activation function");

    // Leaky ReLU 함수 바인딩
    m.def("leaky_relu", &leaky_relu, py::arg("inputs"), py::arg("alpha") = 0.01, "Leaky ReLU activation function");

    // Softmax 함수 바인딩
    m.def("softmax", &softmax, py::arg("inputs"), "Softmax activation function");
}
