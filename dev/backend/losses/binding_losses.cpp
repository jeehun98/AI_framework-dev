// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "losses.cpp"  // 해당 부분을 실제 파일 이름으로 바꿔주세요.

namespace py = pybind11;

PYBIND11_MODULE(losses, m) {
    m.def("mean_squared_error", &mean_squared_error, "Calculate Mean Squared Error",
          py::arg("y_true"), py::arg("y_pred"));

    m.def("cross_entropy_loss", &cross_entropy_loss, "Calculate Cross-Entropy Loss",
          py::arg("y_true"), py::arg("y_pred"));
}
