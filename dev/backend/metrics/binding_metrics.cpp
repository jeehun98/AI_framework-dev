// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "metrics.cpp"  // 해당 부분을 실제 파일 이름으로 바꿔주세요.

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

    m.def("mse", &mean_squared_error, "Calculate mse",
          py::arg("y_true"), py::arg("y_pred"));
}
