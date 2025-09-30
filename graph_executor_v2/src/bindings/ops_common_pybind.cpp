// src/bindings/ops_common_pybind.cpp
//
// Python module name : graph_executor_v2.ops._ops_common
// Role               : 공용 타입(ActKind, GemmAttrs) 1회 노출 전용
//
// Build tip (CMake):
//   pybind11_add_module(_ops_common MODULE src/bindings/ops_common_pybind.cpp)
//   target_include_directories(_ops_common PRIVATE ${PROJECT_INCLUDE_DIR} ${CMAKE_SOURCE_DIR})
//   set_target_properties(_ops_common PROPERTIES PREFIX "" SUFFIX ".pyd"
//                         LIBRARY_OUTPUT_DIRECTORY ".../python/graph_executor_v2/ops")

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "ai/op_schema.hpp"  // ai::ActKind, ai::GemmAttrs 선언이 들어있는 헤더

// (선택) MSVC에서 Windows.h의 min/max 매크로 충돌 회피가 필요하면
// #ifndef NOMINMAX
// #define NOMINMAX
// #endif

PYBIND11_MODULE(_ops_common, m) {
    // 패키지 메타 (import 경로 깔끔하게)
    m.attr("__package__") = "graph_executor_v2.ops";
    m.doc() = R"(Common types shared across ops:
- ActKind: activation kind enum
- GemmAttrs: GEMM attributes (transpose flags, activation, bias, leaky slope))";

    // -------- enum: ActKind --------
    py::enum_<ai::ActKind>(m, "ActKind", py::arithmetic())
        .value("None",      ai::ActKind::None,      "No activation")
        .value("ReLU",      ai::ActKind::ReLU,      "ReLU")
        .value("LeakyReLU", ai::ActKind::LeakyReLU, "Leaky ReLU (uses GemmAttrs.leaky_slope)")
        .value("GELU",      ai::ActKind::GELU,      "GELU (tanh-approx or impl-defined)")
        .value("Sigmoid",   ai::ActKind::Sigmoid,   "Sigmoid")
        .value("Tanh",      ai::ActKind::Tanh,      "Tanh")
        .export_values();

    // -------- struct: GemmAttrs --------
    py::class_<ai::GemmAttrs>(m, "GemmAttrs")
        .def(py::init<>(), "Default-initialize GEMM attributes.")
        .def_readwrite("trans_a",     &ai::GemmAttrs::trans_a,     "Treat A as transposed")
        .def_readwrite("trans_b",     &ai::GemmAttrs::trans_b,     "Treat B as transposed")
        .def_readwrite("act",         &ai::GemmAttrs::act,         "Activation kind (ActKind)")
        .def_readwrite("with_bias",   &ai::GemmAttrs::with_bias,   "Whether epilogue adds bias")
        .def_readwrite("leaky_slope", &ai::GemmAttrs::leaky_slope, "LeakyReLU slope (default 0.01)")
        // 편의: repr
        .def("__repr__", [](const ai::GemmAttrs& a){
            const char* act_name = "Unknown";
            switch (a.act) {
                case ai::ActKind::None:      act_name = "None"; break;
                case ai::ActKind::ReLU:      act_name = "ReLU"; break;
                case ai::ActKind::LeakyReLU: act_name = "LeakyReLU"; break;
                case ai::ActKind::GELU:      act_name = "GELU"; break;
                case ai::ActKind::Sigmoid:   act_name = "Sigmoid"; break;
                case ai::ActKind::Tanh:      act_name = "Tanh"; break;
            }
            return py::str("GemmAttrs(trans_a={}, trans_b={}, act={}, with_bias={}, leaky_slope={})")
                .format(a.trans_a, a.trans_b, act_name, a.with_bias, a.leaky_slope);
        });

    // (선택) __all__ 제공
    m.attr("__all__") = py::make_tuple("ActKind", "GemmAttrs");
}
