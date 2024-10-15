#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "node.h"  // Node 클래스가 정의된 헤더 파일 포함

namespace py = pybind11;

PYBIND11_MODULE(node, m) {
    py::class_<Node, std::shared_ptr<Node>>(m, "Node")
        .def(py::init<const std::string&, double, double, double, double>(),
             py::arg("operation"), py::arg("input_value"), py::arg("weight_value"), py::arg("output"), py::arg("bias"))
        .def(py::init<const std::string&, double, double, double>(),
             py::arg("operation"), py::arg("input_value"), py::arg("output"), py::arg("bias"))
        
        // 부모 및 자식 노드 관련 메서드
        .def("add_parent", &Node::add_parent)
        .def("add_child", &Node::add_child)
        .def("remove_parent", &Node::remove_parent)
        .def("remove_child", &Node::remove_child)
        
        // 노드 업데이트 메서드
        .def("update", &Node::update)
        
        // 자식 및 부모 노드 반환, std::shared_ptr 처리를 위해 reference_internal 사용
        .def("get_children", &Node::get_children, py::return_value_policy::reference_internal)
        .def("get_parents", &Node::get_parents, py::return_value_policy::reference_internal)

        // 리프 노드를 찾는 메서드 추가
        .def("find_leaf_nodes", &Node::find_leaf_nodes, py::return_value_policy::reference_internal)
        
        // 그래디언트 계산 메서드
        .def("calculate_gradient", &Node::calculate_gradient)
        
        // 역전파 수행
        .def("backpropagate", &Node::backpropagate)
        
        // 가중치 업데이트 (unordered_set을 처리하여 중복 방문 방지)
        .def("update_weights", [](Node& self, double learning_rate) {
            std::unordered_set<Node*> visited;
            self.update_weights(learning_rate, &visited);
        }, py::arg("learning_rate"))
        
        // 속성 직접 접근 허용
        .def_readwrite("operation", &Node::operation)
        .def_readwrite("input_value", &Node::input_value)
        .def_readwrite("weight_value", &Node::weight_value)
        .def_readwrite("output", &Node::output)
        .def_readwrite("bias", &Node::bias)
        .def_readwrite("grad_bias", &Node::grad_bias)
        .def_readwrite("grad_weight_total", &Node::grad_weight_total);
}
