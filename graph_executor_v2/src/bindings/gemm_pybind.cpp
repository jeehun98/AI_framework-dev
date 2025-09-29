// src/bindings/gemm_pybind.cpp
//
// Build target name: _ops_gemm
// Output location  : python/graph_executor_v2/ops/_ops_gemm[.pyd|.so]
//
// Requirements:
//   - ai::Status, ai::StreamHandle  : "ai/dispatch.hpp"
//   - ai::Tensor, ai::GemmAttrs     : "ai/tensor.hpp", "ai/op_schema.hpp"
//   - GEMM FWD/BWD API              : "backends/cuda/ops/gemm/api.hpp"
// Notes:
//   - Status -> Python 예외 변환
//   - 모든 출력/옵셔널 인자는 None 허용 (nullptr로 전달)
//   - stream 인자는 void* (cudaStream_t reinterpret_cast)
//   - 현재 Tensor/GemmAttrs가 파이썬에 노출되지 않은 환경에서도 쓸 수 있도록
//     NumPy 친화 오버로드(forward_numpy/backward_numpy)를 함께 제공

#include <string>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>      // for py::array_t
// 필요 시: #include <pybind11/stl.h>

#include "backends/cuda/ops/_common/shim/ai_shim.hpp"

#include "backends/cuda/ops/gemm/api.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

// -------- Status -> Python 예외 --------
static void raise_if_not_ok(ai::Status st, const char* where) {
    using S = ai::Status;
    if (st == S::Ok) return;

    const char* msg = "GEMM op failed";
    switch (st) {
        case S::DeviceMismatch:        msg = "DeviceMismatch: not CUDA tensor or device mismatch"; break;
        case S::DtypeMismatch:         msg = "DtypeMismatch: only f32 is supported currently"; break;
        case S::LayoutMismatch:        msg = "LayoutMismatch: only row-major is supported"; break;
        case S::TransposeNotSupported: msg = "TransposeNotSupported: only non-transposed path supported"; break;
        case S::ShapeMismatch:         msg = "ShapeMismatch: check M,K / K,N / Y(M,N)"; break;
        case S::StrideMismatch:        msg = "StrideMismatch: invalid leading dimensions (lda/ldb/ldd)"; break;
        case S::MissingInput:          msg = "MissingInput: required input missing (e.g., C for gC)"; break;
        case S::Invalid:               msg = "Invalid: argument range/validity error (e.g., int32 overflow)"; break;
        default:                       msg = "Unknown error";
    }
    throw std::runtime_error(std::string("[_ops_gemm::") + where + "] " + msg);
}

PYBIND11_MODULE(_ops_gemm, m) {
    m.doc() = R"(graph_executor_v2 GEMM bindings (regemm epilogue: bias+activation fused)
- forward/backward: f32, row-major, no-transpose
- bias broadcasting priority: Scalar > PerN(len==N) > PerM(len==M)
- If ai::Tensor/GemmAttrs are not exposed to Python, use forward_numpy/backward_numpy.)";

    // =========================
    // 1) 저수준 (ai::Tensor 기반)
    // =========================

    // forward: A:[M,K], B:[K,N], Bias:[1|M|N]|None -> Y:[M,N]
    m.def(
        "forward",
        [](const ai::Tensor& A,
           const ai::Tensor& B,
           const ai::Tensor* Bias,
           ai::Tensor& Y,
           const ai::GemmAttrs& attrs,
           void* stream /* = nullptr */) {
            auto st = ai::GemmCudaLaunch(A, B, Bias, Y, attrs, stream);
            raise_if_not_ok(st, "forward");
        },
        py::arg("A"),
        py::arg("B"),
        py::arg("bias") = nullptr,
        py::arg("Y"),
        py::arg("attrs"),
        py::arg("stream") = nullptr,
        "Run fused GEMM(+bias+activation) on CUDA (low-level, ai::Tensor based)."
    );

    // backward:
    // Inputs : A:[M,K], B:[K,N], C:[M,N]|None, gY:[M,N], Z:[M,N]
    // Outputs: gA:[M,K]|None, gB:[K,N]|None, gC:[M,N]|None, gBias:[1|M|N]|None
    m.def(
        "backward",
        [](const ai::Tensor& A,
           const ai::Tensor& B,
           const ai::Tensor* C,      // optional
           const ai::Tensor& gY,
           const ai::Tensor& Z,
           ai::Tensor* gA,           // optional
           ai::Tensor* gB,           // optional
           ai::Tensor* gC,           // optional
           ai::Tensor* gBias,        // optional
           const ai::GemmAttrs& attrs,
           void* stream /* = nullptr */) {
            auto st = ai::GemmCudaBackward(
                A, B, C, gY, Z, gA, gB, gC, gBias, attrs, stream
            );
            raise_if_not_ok(st, "backward");
        },
        py::arg("A"),
        py::arg("B"),
        py::arg("C")     = nullptr,
        py::arg("gY"),
        py::arg("Z"),
        py::arg("gA")    = nullptr,
        py::arg("gB")    = nullptr,
        py::arg("gC")    = nullptr,
        py::arg("gBias") = nullptr,
        py::arg("attrs"),
        py::arg("stream") = nullptr,
        "Compute gradients for fused GEMM(+bias+activation) (low-level, ai::Tensor based)."
    );

    // ===========================================
    // 2) NumPy 친화 오버로드 (현재 구조에서 바로 사용)
    //    내부적으로 graph_executor_v2._core 함수형 API 호출
    // ===========================================
    m.def(
        "forward_numpy",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> A,
           py::array_t<float, py::array::c_style | py::array::forcecast> B,
           py::object bias,                         // None or 1D float array
           std::string act,
           float leaky_slope) {
            // _core import & call: gemm_bias_act(A, B, bias, act=..., leaky_slope=...)
            py::module_ core = py::module_::import("graph_executor_v2._core");
            py::object fn = core.attr("gemm_bias_act");
            py::object bias_arg = bias.is_none() ? py::none() : bias;
            // kwargs 전달
            py::object Y = fn(A, B, bias_arg, "act"_a = act, "leaky_slope"_a = leaky_slope);
            return Y; // py::array
        },
        py::arg("A"),
        py::arg("B"),
        py::arg("bias") = py::none(),
        py::arg("act") = "none",
        py::arg("leaky_slope") = 0.01f,
        R"(Convenience wrapper that accepts NumPy arrays and delegates to graph_executor_v2._core.gemm_bias_act.)"
    );

    m.def(
        "backward_numpy",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> A,
           py::array_t<float, py::array::c_style | py::array::forcecast> B,
           py::array_t<float, py::array::c_style | py::array::forcecast> gY,
           py::array_t<float, py::array::c_style | py::array::forcecast> Z,
           std::string act,
           std::string bias_kind,
           float leaky_slope) {
            // _core import & call: gemm_bias_act_bwd(A, B, gY, Z, act=..., bias_kind=..., leaky_slope=...)
            py::module_ core = py::module_::import("graph_executor_v2._core");
            py::object fn = core.attr("gemm_bias_act_bwd");
            py::dict out = fn(A, B, gY, Z,
                              "act"_a = act,
                              "bias_kind"_a = bias_kind,
                              "leaky_slope"_a = leaky_slope).cast<py::dict>();
            return out; // dict with gA,gB,(gBias)
        },
        py::arg("A"),
        py::arg("B"),
        py::arg("gY"),
        py::arg("Z"),
        py::arg("act") = "none",
        py::arg("bias_kind") = "none",
        py::arg("leaky_slope") = 0.01f,
        R"(Convenience wrapper that accepts NumPy arrays and delegates to graph_executor_v2._core.gemm_bias_act_bwd.)"
    );

    // (선택) 모듈 메타
    m.attr("__package__") = "graph_executor_v2.ops";
    m.attr("__all__")     = py::make_tuple("forward", "backward", "forward_numpy", "backward_numpy");
}
