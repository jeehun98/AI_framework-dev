#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "operations_matrix.cpp"
#include "../node/node.h"

namespace py = pybind11;

PYBIND11_MODULE(operations_matrix, m) {
    m.doc() = "Matrix operations with computation graph support";

    m.def("matrix_add", &matrix_add, 
          py::arg("A"), py::arg("B"), py::arg("node_list") = std::vector<std::shared_ptr<Node>>(), 
          "Matrix addition with optional node_list");

    m.def("matrix_multiply", &matrix_multiply, 
          py::arg("A"), py::arg("B"), py::arg("node_list") = std::vector<std::shared_ptr<Node>>(), 
          "Matrix multiplication with optional node_list");
}
