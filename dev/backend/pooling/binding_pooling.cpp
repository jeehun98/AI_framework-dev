#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include "../node/node.h"  // Node 클래스가 정의된 헤더 파일 포함
#include "pooling.cpp"

PYBIND11_MODULE(pooling, m) {
    m.def("pooling2d", &pooling2d, "2D Pooling operation (Max/Avg)",
          py::arg("input"), py::arg("pool_height"), py::arg("pool_width"),
          py::arg("stride") = 1, 
          py::arg("mode") = "max", 
          py::arg("node_list") = std::vector<std::shared_ptr<Node>>());
}