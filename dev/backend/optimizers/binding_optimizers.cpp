#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "optimizers.cpp"  // 실제 파일 이름으로 변경하세요. 보통 헤더 파일로 변경 권장.
#include "../node/node.h"  // Node 클래스 헤더 파일 경로

namespace py = pybind11;

PYBIND11_MODULE(optimizers, m) {
    py::class_<SGD>(m, "SGD")
        .def(py::init<double>(), py::arg("learning_rate"))
        .def("update", &SGD::update, py::arg("node"))  // 바인딩에 문제가 없는지 확인
        .def("update_all_weights", &SGD::update_all_weights, py::arg("root"));  // 올바른 메서드가 바인딩되어 있는지 확인

    py::class_<Adam>(m, "Adam")
        .def(py::init<double, double, double, double>(),
             py::arg("learning_rate"), py::arg("beta1") = 0.9, py::arg("beta2") = 0.999, py::arg("epsilon") = 1e-8)
        .def("update", &Adam::update, py::arg("node"))  // 바인딩에 문제가 없는지 확인
        .def("update_all_weights", &Adam::update_all_weights, py::arg("root"));  // 올바른 메서드가 바인딩되어 있는지 확인
}
