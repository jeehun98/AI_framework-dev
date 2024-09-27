#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include "../node/node.h"  // Node 클래스가 정의된 헤더 파일 포함
#include "convolution.cpp"

namespace py = pybind11;

// Python 바인딩 정의
PYBIND11_MODULE(backend, m) {
    m.doc() = "Convolution module using C++ backend with pybind11";  // 모듈 설명

    // conv2d 함수 바인딩
    m.def("conv2d", &conv2d, 
          py::arg("input"), 
          py::arg("filters"), 
          py::arg("stride") = 1, 
          py::arg("padding") = "valid", 
          py::arg("node_list") = std::vector<std::shared_ptr<Node>>{},
          "Perform 2D convolution with optional padding and stride. "
          "Returns the result and the node list.");
}
