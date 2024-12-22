#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "node.h" // Node 클래스가 정의된 헤더 파일 포함

namespace py = pybind11;

PYBIND11_MODULE(node, m) {
    py::class_<Node, std::shared_ptr<Node>>(m, "Node")
        .def(py::init<const std::string&, double, double, double, double>(),
             py::arg("operation"), py::arg("input_value"), py::arg("weight_value"), py::arg("output"), py::arg("bias"))
        .def(py::init<const std::string&, double, double, double>(),
             py::arg("operation"), py::arg("input_value"), py::arg("output"), py::arg("bias"))

        .def("add_parent", &Node::add_parent)
        .def("add_child", &Node::add_child)

        .def("remove_parent", [](Node& self, std::shared_ptr<Node> parent) {
            auto parents = self.get_parents();
            auto it = std::remove(parents.begin(), parents.end(), parent);
            if (it != parents.end()) {
                parents.erase(it, parents.end());
            }
        })
        .def("remove_child", [](Node& self, std::shared_ptr<Node> child) {
            auto children = self.get_children();
            auto it = std::remove(children.begin(), children.end(), child);
            if (it != children.end()) {
                children.erase(it, children.end());
            }
        })

        .def("update", &Node::update)
        .def("get_children", [](Node& self) {
            return py::cast(self.get_children());
        }, py::return_value_policy::reference_internal)
        .def("get_parents", [](Node& self) {
            return py::cast(self.get_parents());
        }, py::return_value_policy::reference_internal)

        .def("find_leaf_nodes", [](Node& self) {
            return py::cast(self.find_leaf_nodes());
        }, py::return_value_policy::reference_internal)

        .def("calculate_gradient", &Node::calculate_gradient)

        .def("backpropagate", [](Node& self, double upstream_gradient = 1.0) {
            std::unordered_set<Node*> visited;
            self.backpropagate(upstream_gradient, &visited);
        }, py::arg("upstream_gradient") = 1.0)

        .def("update_weights", [](Node& self, double learning_rate) {
            std::unordered_set<Node*> visited;
            self.update_weights(learning_rate, &visited);
        }, py::arg("learning_rate"))

        .def("print_tree", [](std::shared_ptr<Node> self) {
            self->print_tree(self);
        })

        .def_readwrite("operation", &Node::operation)
        .def_readwrite("input_value", &Node::input_value)
        .def_readwrite("weight_value", &Node::weight_value)
        .def_readwrite("output", &Node::output)
        .def_readwrite("bias", &Node::bias)
        .def_readwrite("grad_weight_total", &Node::grad_weight_total);
}
