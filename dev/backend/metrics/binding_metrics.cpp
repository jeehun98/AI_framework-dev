// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "metrics.cpp"

namespace py = pybind11;

PYBIND11_MODULE(metrics, m) {
    m.def("accuracy", &accuracy, "Calculate accuracy",
          py::arg("y_true"), py::arg("y_pred"));
    
    m.def("precision", &precision, "Calculate precision",
          py::arg("y_true"), py::arg("y_pred"));
    
    m.def("recall", &recall, "Calculate recall",
          py::arg("y_true"), py::arg("y_pred"));
    
    m.def("f1_score", &f1_score, "Calculate F1 score",
          py::arg("y_true"), py::arg("y_pred"));
}
