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
//   - Tensor/GemmAttrs를 직접 쓰거나, NumPy 친화 오버로드 사용 가능
//   - 공용 타입은 graph_executor_v2.ops._ops_common 에서 1회 노출됨

#include <string>
#include <stdexcept>
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

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

// -------- 문자열 -> ActKind 파서 --------
static ai::ActKind parse_act(const std::string& s) {
    std::string k(s);
    std::transform(k.begin(), k.end(), k.begin(), ::tolower);
    if (k=="none")       return ai::ActKind::None;
    if (k=="relu")       return ai::ActKind::ReLU;
    if (k=="leakyrelu" || k=="leaky_relu" || k=="lrelu")
                         return ai::ActKind::LeakyReLU;
    if (k=="gelu")       return ai::ActKind::GELU;
    if (k=="sigmoid")    return ai::ActKind::Sigmoid;
    if (k=="tanh")       return ai::ActKind::Tanh;
    throw std::invalid_argument("unknown act: " + s);
}

PYBIND11_MODULE(_ops_gemm, m) {
    // 공용 타입(ActKind, GemmAttrs) 등록 모듈 선 import
    // (CMake에서 add_dependencies(_ops_gemm _ops_common)도 함께 설정)
    py::module_::import("graph_executor_v2.ops._ops_common");

    m.doc() = R"(graph_executor_v2 GEMM bindings (regemm epilogue: bias+activation fused)
- forward/backward: f32, row-major, no-transpose
- bias broadcasting priority: Scalar > PerN(len==N) > PerM(len==M)
- Use GemmAttrs directly or forward_ex/backward_ex with Python primitives.
- NumPy helpers delegate to graph_executor_v2._core.)";

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
        "Run fused GEMM(+bias+activation) on CUDA (low-level, ai::Tensor + GemmAttrs)."
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
        "Compute gradients for fused GEMM(+bias+activation) (low-level, ai::Tensor + GemmAttrs)."
    );

    // =========================
    // 1-EX) 편의 오버로드: attrs 없이 호출
    // =========================
    m.def(
        "forward_ex",
        [](const ai::Tensor& A,
           const ai::Tensor& B,
           const ai::Tensor* Bias,
           ai::Tensor& Y,
           bool trans_a,
           bool trans_b,
           std::string act,
           bool with_bias,
           float leaky_slope,
           void* stream /*=nullptr*/) {
            ai::GemmAttrs attrs{};
            attrs.trans_a     = trans_a;
            attrs.trans_b     = trans_b;
            attrs.act         = parse_act(act);
            attrs.with_bias   = with_bias;
            attrs.leaky_slope = leaky_slope;
            if (attrs.with_bias && Bias == nullptr) {
                throw std::invalid_argument("with_bias=True but bias is None");
            }
            auto st = ai::GemmCudaLaunch(A, B, Bias, Y, attrs, stream);
            raise_if_not_ok(st, "forward_ex");
        },
        py::arg("A"),
        py::arg("B"),
        py::arg("bias") = nullptr,
        py::arg("Y"),
        py::arg("trans_a") = false,
        py::arg("trans_b") = false,
        py::arg("act") = "none",
        py::arg("with_bias") = false,
        py::arg("leaky_slope") = 0.01f,
        py::arg("stream") = nullptr,
        "Run fused GEMM with epilogue params provided as Python primitives."
    );

    m.def(
        "backward_ex",
        [](const ai::Tensor& A,
           const ai::Tensor& B,
           const ai::Tensor* C,
           const ai::Tensor& gY,
           const ai::Tensor& Z,
           ai::Tensor* gA,
           ai::Tensor* gB,
           ai::Tensor* gC,
           ai::Tensor* gBias,
           bool trans_a,
           bool trans_b,
           std::string act,
           bool with_bias,
           float leaky_slope,
           void* stream /*=nullptr*/) {
            ai::GemmAttrs attrs{};
            attrs.trans_a     = trans_a;
            attrs.trans_b     = trans_b;
            attrs.act         = parse_act(act);
            attrs.with_bias   = with_bias;
            attrs.leaky_slope = leaky_slope;
            auto st = ai::GemmCudaBackward(A, B, C, gY, Z, gA, gB, gC, gBias, attrs, stream);
            raise_if_not_ok(st, "backward_ex");
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
        py::arg("trans_a") = false,
        py::arg("trans_b") = false,
        py::arg("act") = "none",
        py::arg("with_bias") = false,
        py::arg("leaky_slope") = 0.01f,
        py::arg("stream") = nullptr,
        "Backward for fused GEMM with epilogue params provided as Python primitives."
    );

    // ===========================================
    // 2) NumPy 친화 오버로드 (상위 _core 위임)
    // ===========================================
    m.def(
        "forward_numpy",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> A,
           py::array_t<float, py::array::c_style | py::array::forcecast> B,
           py::object bias, // None or 1D float array
           std::string act,
           float leaky_slope) {
            py::module_ core = py::module_::import("graph_executor_v2._core");
            py::object fn = core.attr("gemm_bias_act");
            py::object bias_arg = bias.is_none() ? py::none() : bias;
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

    // 메타
    m.attr("__package__") = "graph_executor_v2.ops";
    m.attr("__all__")     = py::make_tuple(
        "forward", "backward",
        "forward_ex", "backward_ex",
        "forward_numpy", "backward_numpy"
    );
}
