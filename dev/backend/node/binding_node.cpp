#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Node.h"

namespace py = pybind11;

PYBIND11_MODULE(node, m) {
    py::class_<Node, std::shared_ptr<Node>>(m, "Node")
        .def(py::init<const std::string&, double, double, double, double>(),
             py::arg("operation"), py::arg("input_value"), py::arg("weight_value"), py::arg("output"), py::arg("bias"))
        .def(py::init<const std::string&, double, double, double>(),
             py::arg("operation"), py::arg("input_value"), py::arg("output"), py::arg("bias"))
        .def("add_parent", &Node::add_parent)
        .def("add_child", &Node::add_child)
        .def("remove_parent", &Node::remove_parent)
        .def("remove_child", &Node::remove_child)
        .def("update", &Node::update)
        .def("get_children", &Node::get_children)
        .def("get_parents", &Node::get_parents)
        .def("calculate_gradient", &Node::calculate_gradient)
        .def("backpropagate", &Node::backpropagate)
        .def("update_weights",
             [](Node& self, double learning_rate) {
                 std::unordered_set<Node*> visited;
                 self.update_weights(learning_rate, &visited);
             },
             py::arg("learning_rate"))
        // 멤버 변수 노출
        .def_readwrite("operation", &Node::operation)
        .def_readwrite("input_value", &Node::input_value)
        .def_readwrite("weight_value", &Node::weight_value)
        .def_readwrite("output", &Node::output)
        .def_readwrite("bias", &Node::bias)
        .def_readwrite("grad_bias", &Node::grad_bias)
        .def_readwrite("grad_weight_total", &Node::grad_weight_total);  // grad_weight_total 노출 추가
}
