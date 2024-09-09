#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "operations_matrix.cpp"
#include "../node/node.h"

namespace py = pybind11;

PYBIND11_MODULE(operations_matrix, m) {

    m.def("matrix_add", &matrix_add, py::arg("A"), py::arg("B"));
    m.def("matrix_multiply", &matrix_multiply, py::arg("A"), py::arg("B"), py::arg("node_list") = std::vector<std::shared_ptr<Node>>());
}