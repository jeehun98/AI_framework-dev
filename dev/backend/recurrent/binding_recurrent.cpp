#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <memory>
#include "../node/node.h"  // Node 클래스 헤더 파일
#include "recurrent.cpp"

namespace py = pybind11;

// Pybind11 바인딩
PYBIND11_MODULE(rnn, m) {
    // rnn_layer 함수 바인딩
    m.def("rnn_layer", &rnn_layer, 
          py::arg("input"), 
          py::arg("weights"), 
          py::arg("recurrent_weights"), 
          py::arg("bias"), 
          py::arg("activation") = "tanh", 
          py::arg("node_list") = std::vector<std::shared_ptr<Node>>{},
          "Perform RNN layer computation with optional activation and node tracking. "
          "Returns the result and the updated node list.");
}
