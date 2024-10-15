// operations_matrix.h
#ifndef OPERATIONS_MATRIX_H
#define OPERATIONS_MATRIX_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <memory>
#include "../node/node.h" // Node 클래스가 정의된 헤더 파일 포함

namespace py = pybind11;

std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> matrix_add(
    py::array_t<double> A, 
    py::array_t<double> B,
    std::vector<std::shared_ptr<Node>> node_list = {}
);

std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> matrix_multiply(
    py::array_t<double> A, 
    py::array_t<double> B, 
    std::vector<std::shared_ptr<Node>> node_list = {}
);

#endif // OPERATIONS_MATRIX_H
