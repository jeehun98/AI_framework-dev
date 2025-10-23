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
//   - B안: 여기서 _ops_common의 타입들을 re-export 하여 편의 제공

#include <string>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cctype>        // <-- tolower 안전 사용

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#include "backends/cuda/ops/gemm/api.hpp"

// NVTX shim: USE_NVTX가 켜지면 nvToolsExt를 통해 활성, 아니면 컴파일 타임 no-op
#include "backends/cuda/ops/_common/shim/nvtx.hpp"
#include "backends/cuda/ops/gemm/detail/nvtx_shim.h" // ← 추가 (NVTX_COLOR, NVTX_MARK 제공)



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
        case S::MissingOutput:         msg = "MissingOutput: required output missing (e.g., save_z=True but Z_saved is None)"; break;
        case S::Invalid:               msg = "Invalid: argument range/validity error (e.g., int32 overflow)"; break;
        default:                       msg = "Unknown error";
    }
    throw std::runtime_error(std::string("[_ops_gemm::") + where + "] " + msg);
}

// -------- 문자열 -> ActKind 파서 --------
static ai::ActKind parse_act(const std::string& s) {
    std::string k(s);
    std::transform(k.begin(), k.end(), k.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
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
    m.attr("__package__") = "graph_executor_v2.ops";
    m.doc() = R"(graph_executor_v2 GEMM bindings (regemm epilogue: bias+activation fused)
- forward/backward: f32, row-major, no-transpose
- bias broadcasting priority: Scalar > PerN(len==N) > PerM(len==M)
- Supports saving pre-activation Z in forward (save_z).
- NumPy helpers delegate to graph_executor_v2._core.)";

    // ======================================================
    // 공용 타입 모듈 import + re-export (B안)
    // ======================================================
    py::module_ common = py::module_::import("graph_executor_v2.ops._ops_common");
    m.attr("ActKind")        = common.attr("ActKind");
    m.attr("GemmAttrs")      = common.attr("GemmAttrs");
    m.attr("Device")         = common.attr("Device");
    m.attr("DType")          = common.attr("DType");
    m.attr("Layout")         = common.attr("Layout");
    m.attr("TensorDesc")     = common.attr("TensorDesc");
    m.attr("Tensor")         = common.attr("Tensor");
    m.attr("make_tensor_2d") = common.attr("make_tensor_2d");

    // =========================
    // 1) 저수준 (ai::Tensor 기반)
    // =========================

    // forward: A:[M,K], B:[K,N], Bias:[1|M|N]|None -> Y:[M,N]
    // Optional: Z_saved:[M,N] to stash pre-activation (attrs.save_z True or implied)
    m.def(
        "forward",
        [](const ai::Tensor& A,
           const ai::Tensor& B,
           const ai::Tensor* Bias,
           ai::Tensor& Y,
           ai::GemmAttrs attrs,
           ai::Tensor* Z_saved,          // optional
           void* stream /* = nullptr */) {
            py::gil_scoped_release release; // 🔓 GIL off (네이티브 CUDA 실행)
            // If Z_saved is provided but attrs.save_z is false, enable it implicitly
            if (Z_saved && Z_saved->data && !attrs.save_z) {
                attrs.save_z = true;
            }
            // If attrs.save_z is true but Z_saved is null -> raise early for clearer error
            if (attrs.save_z && (!Z_saved || !Z_saved->data)) {
                throw std::invalid_argument(
                    "[_ops_gemm::forward] save_z=True requires a valid Z_saved Tensor");
            }
            NVTX_RANGE("gemm.forward", NVTX_COLOR::Orange);
            
            auto st = ai::GemmCudaLaunch(A, B, Bias, Y, attrs, stream, Z_saved);
            
        },
        py::arg("A"),
        py::arg("B"),
        py::arg("bias") = nullptr,
        py::arg("Y"),
        py::arg("attrs"),
        py::arg("Z_saved") = nullptr,
        py::arg("stream") = nullptr,
        "Run fused GEMM(+bias+activation). Optionally saves pre-activation Z into Z_saved."
    );

    // backward:
    // Inputs : A:[M,K], B:[K,N], C:[M,N]|None, gY:[M,N], Z:[M,N]
    // Outputs: gA:[M,K]|None, gB:[K,N]|None, gC:[M,N]|None, gBias:[1|N]|None
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
            py::gil_scoped_release release; // 🔓 GIL off (네이티브 CUDA 실행)
            // ---- PerN 강제 세이프가드 ----
            const int64_t M = A.desc.shape.at(0);
            const int64_t N = B.desc.shape.at(1);
            if (gBias && gBias->data) {
                const auto& s = gBias->desc.shape;
                const bool bad_perm =
                    (s.size()==1 && s[0]==M) ||
                    (s.size()==2 && s[0]==M && s[1]==1);
                if (bad_perm) {
                    throw std::invalid_argument(
                        "[_ops_gemm::backward] gBias shape suggests PerM (len==M). "
                        "Bias grad for Dense must be PerN; allocate gBias as (1,N) or (N,).");
                }
            }
            NVTX_RANGE("gemm.backward", NVTX_COLOR::Red);
            
            auto st = ai::GemmCudaBackward(
                A, B, C, gY, Z, gA, gB, gC, gBias, attrs, stream
            );
            
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
        "Compute gradients for fused GEMM(+bias+activation) using pre-activation Z."
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
           bool save_z,               // NEW
           ai::Tensor* Z_saved,       // NEW
           void* stream /*=nullptr*/) {
            py::gil_scoped_release release; // 🔓 GIL off (네이티브 CUDA 실행)
            ai::GemmAttrs attrs{};
            attrs.trans_a     = trans_a;
            attrs.trans_b     = trans_b;
            attrs.act         = parse_act(act);
            attrs.with_bias   = with_bias;
            attrs.leaky_slope = leaky_slope;
            attrs.save_z      = save_z;

            if (attrs.with_bias && Bias == nullptr) {
                throw std::invalid_argument("with_bias=True but bias is None");
            }
            if (attrs.save_z) {
                if (!Z_saved || !Z_saved->data) {
                    throw std::invalid_argument("save_z=True requires Z_saved Tensor");
                }
            } else if (Z_saved && Z_saved->data) {
                // User passed Z_saved without setting save_z -> enable implicitly
                attrs.save_z = true;
            }

            NVTX_RANGE("gemm.forward_ex", NVTX_COLOR::Teal);
            
            auto st = ai::GemmCudaLaunch(A, B, Bias, Y, attrs, stream, Z_saved);
            
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
        py::arg("save_z") = false,
        py::arg("Z_saved") = nullptr,
        py::arg("stream") = nullptr,
        "Run fused GEMM with epilogue params provided as Python primitives. "
        "Optionally save pre-activation into Z_saved when save_z=True."
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
            py::gil_scoped_release release; // 🔓 GIL off (네이티브 CUDA 실행)
            ai::GemmAttrs attrs{};
            attrs.trans_a     = trans_a;
            attrs.trans_b     = trans_b;
            attrs.act         = parse_act(act);
            attrs.with_bias   = with_bias;
            attrs.leaky_slope = leaky_slope;

            // ---- PerN 강제 세이프가드 ----
            const int64_t M = A.desc.shape.at(0);
            const int64_t N = B.desc.shape.at(1);
            if (gBias && gBias->data) {
                const auto& s = gBias->desc.shape;
                const bool bad_perm =
                    (s.size()==1 && s[0]==M) ||
                    (s.size()==2 && s[0]==M && s[1]==1);
                if (bad_perm) {
                    throw std::invalid_argument(
                        "[_ops_gemm::backward_ex] gBias shape suggests PerM (len==M). "
                        "Bias grad for Dense must be PerN; allocate gBias as (1,N) or (N,).");
                }
            }

            NVTX_RANGE("gemm.backward_ex", NVTX_COLOR::Magenta);
            
            auto st = ai::GemmCudaBackward(A, B, C, gY, Z, gA, gB, gC, gBias, attrs, stream);
            
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
    // 2) NumPy 친화 오버로드 (상위 _core 위임) — GIL 유지!
    // ===========================================
    m.def(
        "forward_numpy",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> A,
           py::array_t<float, py::array::c_style | py::array::forcecast> B,
           py::object bias, // None or 1D float array
           std::string act,
           float leaky_slope) {
            // GIL 필요: 순수 파이썬 호출
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
            // GIL 필요: 순수 파이썬 호출
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

    // ===========================================
    // 3) Capture-safe backward_into (NO allocations)
    // ===========================================
    m.def(
        "backward_into",
        [](const ai::Tensor& A,
           const ai::Tensor& B,
           const ai::Tensor* C,      // optional
           const ai::Tensor& gY,
           const ai::Tensor& Z,
           ai::Tensor* gA,           // optional
           ai::Tensor* gB,           // optional
           ai::Tensor* gC,           // optional
           ai::Tensor* gBias,        // optional (PerN: (1,N) 권장)
           const ai::GemmAttrs& attrs,
           void* stream,
           // --- workspaces ---
           uintptr_t dZ_ptr,         // required: float[M*N]
           uintptr_t lt_ws_ptr,      // optional: cublasLt workspace
           size_t    lt_ws_bytes) {
            py::gil_scoped_release release; // 🔓 GIL off (네이티브 CUDA 실행)

            // PerN shape 가드 (그대로 유지)
            const int64_t M = A.desc.shape.at(0);
            const int64_t N = B.desc.shape.at(1);
            if (gBias && gBias->data) {
                const auto& s = gBias->desc.shape;
                const bool bad_perm = (s.size()==1 && s[0]==M) || (s.size()==2 && s[0]==M && s[1]==1);
                if (bad_perm) {
                    throw std::invalid_argument(
                        "[_ops_gemm::backward_into] gBias must be PerN (shape (1,N) or (N,))");
                }
            }

            if (!dZ_ptr) {
                throw std::invalid_argument(
                    "[_ops_gemm::backward_into] dZ_ptr is required for capture-safe path");
            }

            // lt workspace ptr/bytes 짝 검증
            if ((lt_ws_ptr == 0) != (lt_ws_bytes == 0)) {
                throw std::invalid_argument(
                    "[_ops_gemm::backward_into] lt_workspace ptr/bytes must be both zero or both non-zero");
            }

            // 통합 워크스페이스 사용
            ai::GemmWorkspace ws{};
            ws.scratch            = reinterpret_cast<void*>(dZ_ptr);     // dZ buffer
            ws.scratch_bytes      = static_cast<size_t>(M * N * sizeof(float)); // 정확히 기입
            ws.lt_workspace       = reinterpret_cast<void*>(lt_ws_ptr);
            ws.lt_workspace_bytes = lt_ws_bytes;

            NVTX_RANGE("gemm.backward_into", NVTX_COLOR::Yellow);
            
            auto st = ai::GemmCudaBackward(
                A, B, C, gY, Z, gA, gB, gC, gBias, attrs, stream, &ws
            );
            
        },
        py::arg("A"), py::arg("B"),
        py::arg("C") = nullptr,
        py::arg("gY"), py::arg("Z"),
        py::arg("gA") = nullptr, py::arg("gB") = nullptr, py::arg("gC") = nullptr, py::arg("gBias") = nullptr,
        py::arg("attrs"),
        py::arg("stream") = nullptr,
        py::arg("dZ_ptr"),
        py::arg("lt_ws_ptr") = static_cast<uintptr_t>(0),
        py::arg("lt_ws_bytes") = static_cast<size_t>(0),
        "Capture-safe GEMM backward that uses preallocated workspaces (no malloc during capture)."
    );

    // === 별칭 (attrs 버전 유지용) ===
    m.def(
        "forward_ex_attrs",
        [](const ai::Tensor& A, const ai::Tensor& B, const ai::Tensor* Bias,
           ai::Tensor& Y, ai::GemmAttrs attrs, ai::Tensor* Z_saved, void* stream) {
            py::gil_scoped_release release;
            if (Z_saved && Z_saved->data && !attrs.save_z) attrs.save_z = true;
            if (attrs.save_z && (!Z_saved || !Z_saved->data))
                throw std::invalid_argument("[_ops_gemm::forward_ex_attrs] save_z=True requires Z_saved");
            auto st = ai::GemmCudaLaunch(A, B, Bias, Y, attrs, stream, Z_saved);
            raise_if_not_ok(st, "forward_ex_attrs");
        },
        "A"_a, "B"_a, "bias"_a = nullptr, "Y"_a,
        "attrs"_a, "Z_saved"_a = nullptr, "stream"_a = nullptr,
        "Alias to `forward` that accepts GemmAttrs (kept for Python compatibility)."
    );

    m.def(
        "backward_ex_attrs",
        [](const ai::Tensor& A, const ai::Tensor& B, const ai::Tensor* C,
           const ai::Tensor& gY, const ai::Tensor& Z,
           ai::Tensor* gA, ai::Tensor* gB, ai::Tensor* gC, ai::Tensor* gBias,
           const ai::GemmAttrs& attrs, void* stream) {
            py::gil_scoped_release release;
            const int64_t M = A.desc.shape.at(0);
            const int64_t N = B.desc.shape.at(1);
            if (gBias && gBias->data) {
                const auto& s = gBias->desc.shape;
                if ((s.size()==1 && s[0]==M) || (s.size()==2 && s[0]==M && s[1]==1))
                    throw std::invalid_argument("[_ops_gemm::backward_ex_attrs] gBias must be PerN (1,N) or (N,)");
            }
            auto st = ai::GemmCudaBackward(A, B, C, gY, Z, gA, gB, gC, gBias, attrs, stream);
            raise_if_not_ok(st, "backward_ex_attrs");
        },
        "A"_a, "B"_a, "C"_a = nullptr, "gY"_a, "Z"_a,
        "gA"_a = nullptr, "gB"_a = nullptr, "gC"_a = nullptr, "gBias"_a = nullptr,
        "attrs"_a, "stream"_a = nullptr,
        "Alias to `backward` that accepts GemmAttrs (kept for Python compatibility)."
    );

    // 메타
    m.attr("__package__") = "graph_executor_v2.ops";
    m.attr("__all__") = py::make_tuple(
        // re-export된 공용 타입들
        "ActKind", "GemmAttrs",
        "Device", "DType", "Layout", "TensorDesc", "Tensor", "make_tensor_2d",
        // 바인딩 함수들
        "forward", "backward",
        "forward_ex", "backward_ex",
        "forward_numpy", "backward_numpy",
        "backward_into",
        "forward_ex_attrs", "backward_ex_attrs"
    );
}
