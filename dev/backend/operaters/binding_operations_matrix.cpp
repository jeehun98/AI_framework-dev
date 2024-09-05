#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "operations_matrix.cpp"
#include "../node/node.h"

namespace py = pybind11;

PYBIND11_MODULE(operations_matrix, m) {
    py::class_<Node, std::shared_ptr<Node>>(m, "Node")
        .def(py::init<const std::string&, double, double, double>())
        .def_readwrite("operation", &Node::operation)
        .def_readwrite("input_a", &Node::input_a)
        .def_readwrite("input_b", &Node::input_b)
        .def_readwrite("output", &Node::output)
        .def_readwrite("parents", &Node::parents)   // 부모 노드 리스트도 바인딩
        .def_readwrite("children", &Node::children); // 자식 노드 리스트도 바인딩

    // 세 번째 인자를 생략하여 함수 바인딩
    m.def("matrix_multiply", 
          [](py::array_t<double> A, py::array_t<double> B) {
              return matrix_multiply(A, B);  // 기본값을 사용하여 세 번째 인자를 생략
          }, 
          "Multiply two matrices", py::arg("A"), py::arg("B"));
}
