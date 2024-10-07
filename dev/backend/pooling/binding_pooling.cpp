#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // tuple, vector 등 STL 타입 지원
#include <vector>
#include "../node/node.h"  // Node 클래스가 정의된 헤더 파일 포함
#include "pooling.cpp"

namespace py = pybind11;

PYBIND11_MODULE(pooling, m) {
    m.def("pooling2d", &pooling2d, "2D Pooling operation (Max/Avg)",
          py::arg("input"),
          py::arg("pool_height"),
          py::arg("pool_width"),
          py::arg("stride") = std::make_pair(1, 1),  // stride를 기본값 (1,1)으로 설정
          py::arg("mode") = "max", 
          py::arg("node_list") = std::vector<std::shared_ptr<Node>>());
}
