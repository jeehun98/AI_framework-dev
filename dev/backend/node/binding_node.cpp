#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Node.h"

namespace py = pybind11;

PYBIND11_MODULE(node, m) {
    py::class_<Node, std::shared_ptr<Node>>(m, "Node")
        // 생성자 정의
        .def(py::init<const std::string&, double, double, double, double>(),
             py::arg("operation"), py::arg("input_value"), py::arg("weight_value"), py::arg("output"), py::arg("bias"))
        .def(py::init<const std::string&, double, double, double>(),
             py::arg("operation"), py::arg("input_value"), py::arg("output"), py::arg("bias"))
        // 메서드 정의
        .def("add_parent", &Node::add_parent, R"pbdoc(
            Adds a parent node to this node.
            
            Parameters:
                parent (Node): The parent node to be added.
        )pbdoc")
        .def("add_child", &Node::add_child, R"pbdoc(
            Adds a child node to this node.
            
            Parameters:
                child (Node): The child node to be added.
        )pbdoc")
        .def("remove_parent", &Node::remove_parent, R"pbdoc(
            Removes a parent node from this node.
            
            Parameters:
                parent (Node): The parent node to be removed.
        )pbdoc")
        .def("remove_child", &Node::remove_child, R"pbdoc(
            Removes a child node from this node.
            
            Parameters:
                child (Node): The child node to be removed.
        )pbdoc")
        .def("update", &Node::update, R"pbdoc(
            Updates the node's input, weight, output, and bias values.
            
            Parameters:
                input (float): The input value for the node.
                weight (float): The weight value for the node.
                output (float): The output value for the node.
                new_bias (float): The new bias value for the node.
        )pbdoc")
        .def("get_children", &Node::get_children, R"pbdoc(
            Returns a list of children nodes.
            
            Returns:
                List[Node]: The list of children nodes.
        )pbdoc")
        .def("get_parents", &Node::get_parents, R"pbdoc(
            Returns a list of parent nodes.
            
            Returns:
                List[Node]: The list of parent nodes.
        )pbdoc")
        .def("calculate_gradient", &Node::calculate_gradient, R"pbdoc(
            Calculates the gradient of the node.
            
            Parameters:
                upstream_gradient (float): The upstream gradient value. Default is 1.0.
            
            Returns:
                Tuple[float, float]: A tuple containing the gradient with respect to input and weight.
        )pbdoc")
        .def("backpropagate", &Node::backpropagate, py::arg("upstream_gradient") = 1.0, R"pbdoc(
            Performs backpropagation starting from this node.
            
            Parameters:
                upstream_gradient (float): The gradient value from the upstream node. Default is 1.0.
        )pbdoc")
        // 수정된 update_weights 메서드 바인딩
        .def("update_weights",
             [](Node& self, double learning_rate) {
                 self.update_weights(learning_rate);  // 기본값으로 호출
             },
             py::arg("learning_rate"), R"pbdoc(
             Updates the weights of the node and its children recursively.
             
             Parameters:
                 learning_rate (float): The learning rate for weight update.
        )pbdoc")
        // 멤버 변수 노출
        .def_readwrite("operation", &Node::operation)
        .def_readwrite("input_value", &Node::input_value)
        .def_readwrite("weight_value", &Node::weight_value)
        .def_readwrite("output", &Node::output)
        .def_readwrite("bias", &Node::bias)
        .def_readwrite("grad_bias", &Node::grad_bias);
}
