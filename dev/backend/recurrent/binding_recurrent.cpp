#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <memory>
#include <string>
#include "recurrent.cpp"  // rnn_layer 함수가 정의된 파일 포함

namespace py = pybind11;

PYBIND11_MODULE(recurrent, m) {
    m.doc() = "RNN layer bindings for Python";  // 모듈 설명

    m.def("rnn_layer", &rnn_layer,
          py::arg("input"),
          py::arg("weights"),
          py::arg("recurrent_weights"),
          py::arg("bias"),
          py::arg("activation"),
          py::arg("node_list") = std::vector<std::shared_ptr<Node>>(),
          "Run an RNN layer computation with specified activation function.\n\n"
          "Parameters:\n"
          "  input: Input sequence data (2D array).\n"
          "  weights: Weights for input data (2D array).\n"
          "  recurrent_weights: Weights for recurrent connections (2D array).\n"
          "  bias: Bias for the layer (1D array).\n"
          "  activation: Activation function name (string).\n"
          "  node_list: List of nodes for the computation graph (optional).\n\n"
          "Returns:\n"
          "  Tuple containing output array and updated node list."
    );
}
