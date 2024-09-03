// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "operations_matrix.cpp"  // 해당 부분을 실제 파일 이름으로 바꿔주세요.

namespace py = pybind11;

PYBIND11_MODULE(operations_matrix, m) {
    m.def("matrix_add", &matrix_add, "Add two matrices",
          py::arg("A"), py::arg("B"));
    
    m.def("matrix_multiply", &matrix_multiply, "Multiply two matrices",
          py::arg("A"), py::arg("B"));
}
