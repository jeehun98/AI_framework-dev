#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <memory>
#include "node.h"

namespace py = pybind11;

PYBIND11_MODULE(node, m) {
    py::class_<Node, std::shared_ptr<Node>>(m, "Node")
        .def(py::init<const std::string&, double, double, double>(),
             py::arg("operation"), py::arg("input_a"), py::arg("input_b"), py::arg("output"))
        .def(py::init<const std::string&, double, double>(),
             py::arg("operation"), py::arg("input_a"), py::arg("output"))
        .def("add_parent", &Node::add_parent)
        .def("add_child", &Node::add_child)
        .def("calculate_gradient", &Node::calculate_gradient)
        .def("get_children", &Node::get_children)
        .def("get_parents", &Node::get_parents)
        .def_readwrite("operation", &Node::operation)
        .def_readwrite("input_a", &Node::input_a)
        .def_readwrite("input_b", &Node::input_b)
        .def_readwrite("output", &Node::output)
        .def_readwrite("grad_input", &Node::grad_input)  // 변경된 변수명 반영
        .def_readwrite("grad_weight", &Node::grad_weight);  // 변경된 변수명 반영
}