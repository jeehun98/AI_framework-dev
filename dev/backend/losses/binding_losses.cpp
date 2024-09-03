// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "losses.cpp"

namespace py = pybind11;

PYBIND11_MODULE(cost_functions, m) {
    m.def("mean_squared_error", &mean_squared_error, "Calculate Mean Squared Error (MSE)",
          py::arg("y_true"), py::arg("y_pred"));

    m.def("cross_entropy_loss", &cross_entropy_loss, "Calculate Cross-Entropy Loss",
          py::arg("y_true"), py::arg("y_pred"));
}
