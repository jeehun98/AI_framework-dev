#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "losses.cpp" 

namespace py = pybind11;

PYBIND11_MODULE(losses, m) {
    m.doc() = "Loss functions with computation graph support";

    m.def("mean_squared_error", &mean_squared_error, "Calculate Mean Squared Error with optional node_list",
          py::arg("y_true"), py::arg("y_pred"), py::arg("node_list") = std::vector<std::shared_ptr<Node>>());

    m.def("cross_entropy_loss", &cross_entropy_loss, "Calculate Cross-Entropy Loss with optional node_list",
          py::arg("y_true"), py::arg("y_pred"), py::arg("node_list") = std::vector<std::shared_ptr<Node>>());
}
