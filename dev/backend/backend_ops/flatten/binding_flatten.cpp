#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // tuple, vector 등 STL 타입 지원
#include <vector>
#include "../node/node.h"  // Node 클래스가 정의된 헤더 파일 포함
#include "flatten.cpp"

namespace py = pybind11;

PYBIND11_MODULE(flatten, m) {
    m.def("flatten", &flatten, "Flatten operation",
          py::arg("input"),
          py::arg("node_list") = std::vector<std::shared_ptr<Node>>());
}
