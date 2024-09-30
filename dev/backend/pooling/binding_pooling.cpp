#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include "../node/node.h"  // Node 클래스가 정의된 헤더 파일 포함
#include "pooling.cpp"

PYBIND11_MODULE(pooling, m) {
    m.doc() = "Pooling module using C++ backend with pybind11";  // 모듈 설명

    // max_pooling 함수 바인딩
    m.def("max_pooling", &max_pooling, 
          py::arg("input"), 
          py::arg("pool_size") = 2, 
          py::arg("stride") = 2, 
          py::arg("padding") = "valid", 
          py::arg("node_list") = std::vector<std::shared_ptr<Node>>{},
          "Perform 2D max pooling with optional padding and stride. "
          "Returns the result and the node list.");
}
