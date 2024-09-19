#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <memory>
#include "node.h"

namespace py = pybind11;

PYBIND11_MODULE(node, m) {
    py::class_<Node, std::shared_ptr<Node>>(m, "Node")
        .def(py::init<const std::string&, double, double, double, double>(),
             py::arg("operation"), py::arg("input_a"), py::arg("input_b"), py::arg("output"), py::arg("weight"))
        .def(py::init<const std::string&, double, double, double>(),
             py::arg("operation"), py::arg("input_a"), py::arg("output"), py::arg("weight"))
        .def("add_parent", &Node::add_parent)
        .def("add_child", &Node::add_child)
        .def("remove_parent", &Node::remove_parent)
        .def("remove_child", &Node::remove_child)
        .def("update", &Node::update)
        .def("calculate_gradient", &Node::calculate_gradient)
        .def("get_children", &Node::get_children)
        .def("get_parents", &Node::get_parents)
        .def_readwrite("operation", &Node::operation)
        .def_readwrite("input_a", &Node::input_a)
        .def_readwrite("input_b", &Node::input_b)
        .def_readwrite("output", &Node::output)
        .def_readwrite("weight", &Node::weight)
        .def_readwrite("grad_weight", &Node::grad_weight);
}