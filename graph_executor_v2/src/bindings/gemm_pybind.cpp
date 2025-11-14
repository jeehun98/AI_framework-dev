// src/bindings/gemm_pybind.cpp
//
// Build target name: _ops_gemm
// Output location  : python/graph_executor_v2/ops/_ops_gemm[.pyd|.so]
//
// Requirements:
//   - shim umbrella types: "backends/cuda/ops/_common/shim/ai_shim.hpp"
//   - GEMM FWD/BWD API    : "backends/cuda/ops/gemm/api.hpp"
//
// Notes:
//   - Status -> Python 예외 변환
//   - 모든 출력/옵셔널 인자는 None 허용 (nullptr로 전달)
//   - stream 인자는 Capsule(void*)로 전달 (cudaStream_t reinterpret_cast)
//   - 공용 타입은 graph_executor_v2.ops._ops_common 에서 1회 노출됨

#include <string>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cctype>
#include <vector>

#include <pybind11/pybind11.h>

#include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#include "backends/cuda/ops/_common/shim/ai_nvtx.hpp"
#include "backends/cuda/ops/gemm/api.hpp"

namespace shim = ::ai::cuda::shim;
namespace py   = pybind11;
using namespace pybind11::literals;

// -------- Status -> Python 예외 --------
static void raise_if_not_ok(shim::Status st, const char* where) {
    using S = shim::Status;
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
static shim::ActKind parse_act(const std::string& s) {
    std::string k(s);
    std::transform(k.begin(), k.end(), k.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    if (k=="none")       return shim::ActKind::None;
    if (k=="relu")       return shim::ActKind::ReLU;
    if (k=="leakyrelu" || k=="leaky_relu" || k=="lrelu")
                         return shim::ActKind::LeakyReLU;
    if (k=="gelu")       return shim::ActKind::GELU;
    if (k=="sigmoid")    return shim::ActKind::Sigmoid;
    if (k=="tanh")       return shim::ActKind::Tanh;
    throw std::invalid_argument("unknown act: " + s);
}

// -------- py::object → Tensor* / Stream 변환 헬퍼 --------
static const shim::Tensor* as_const_tensor_ptr(const py::object& obj) {
    if (obj.is_none()) return nullptr;
    return &obj.cast<const shim::Tensor&>();
}

static shim::Tensor* as_tensor_ptr(const py::object& obj) {
    if (obj.is_none()) return nullptr;
    return &obj.cast<shim::Tensor&>();
}

static shim::StreamHandle as_stream(const py::object& obj) {
    if (obj.is_none()) return nullptr;
    void* p = PyCapsule_GetPointer(obj.ptr(), nullptr);
    return reinterpret_cast<shim::StreamHandle>(p);
}

// -------- raw pointer → shim::Tensor 헬퍼 (f32, row-major, CUDA) --------
static shim::Tensor make_tensor_2d_raw(uintptr_t ptr, int64_t rows, int64_t cols) {
    void* p = reinterpret_cast<void*>(ptr);
    // ai_tensor.hpp 의 factory 사용
    return shim::make_tensor2d(
        p,
        static_cast<std::int64_t>(rows),
        static_cast<std::int64_t>(cols),
        shim::DType::F32  // dtype 고정
    );
}


PYBIND11_MODULE(_ops_gemm, m) {
    m.attr("__package__") = "graph_executor_v2.ops";
    m.doc() = R"(graph_executor_v2 GEMM bindings (regemm epilogue: bias+activation fused)
- forward/backward: f32, row-major, no-transpose
- bias broadcasting priority: Scalar > PerN(len==N) > PerM(len==M)
- Supports saving pre-activation Z in forward (save_z).)";

    // ======================================================
    // 공용 타입 모듈 import + re-export
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
    // 1) 저수준 (shim::Tensor 기반)
    // =========================

    // forward: A:[M,K], B:[K,N], Bias:[1|M|N]|None -> Y:[M,N]
    // Optional: Z_saved:[M,N] to stash pre-activation (attrs.save_z True or implied)
    m.def(
        "forward",
        [](const shim::Tensor& A,
           const shim::Tensor& B,
           py::object bias_obj,
           shim::Tensor& Y,
           shim::GemmAttrs attrs,
           py::object Z_saved_obj,
           py::object stream_obj) {
            py::gil_scoped_release release; // 🔓 GIL off (네이티브 CUDA 실행)

            const shim::Tensor* Bias   = as_const_tensor_ptr(bias_obj);
            shim::Tensor*       Z_saved = as_tensor_ptr(Z_saved_obj);
            shim::StreamHandle  s       = as_stream(stream_obj);

            if (Z_saved && Z_saved->data && !attrs.save_z) {
                attrs.save_z = true;
            }
            if (attrs.save_z && (!Z_saved || !Z_saved->data)) {
                throw std::invalid_argument(
                    "[_ops_gemm::forward] save_z=True requires a valid Z_saved Tensor");
            }

            NVTX_RANGE("gemm.forward", shim::nvtx::Color::Orange);

            auto st = shim::GemmCudaLaunch(A, B, Bias, Y, attrs, s, Z_saved);
            raise_if_not_ok(st, "forward");
        },
        "A"_a,
        "B"_a,
        "bias"_a    = py::none(),
        "Y"_a,
        "attrs"_a,
        "Z_saved"_a = py::none(),
        "stream"_a  = py::none(),
        "Run fused GEMM(+bias+activation). Optionally saves pre-activation Z into Z_saved."
    );

    // backward:
    // Inputs : A:[M,K], B:[K,N], C:[M,N]|None, gY:[M,N], Z:[M,N]
    // Outputs: gA:[M,K]|None, gB:[K,N]|None, gC:[M,N]|None, gBias:[1|N]|None
    m.def(
        "backward",
        [](const shim::Tensor& A,
           const shim::Tensor& B,
           py::object C_obj,
           const shim::Tensor& gY,
           const shim::Tensor& Z,
           py::object gA_obj,
           py::object gB_obj,
           py::object gC_obj,
           py::object gBias_obj,
           const shim::GemmAttrs& attrs,
           py::object stream_obj) {
            py::gil_scoped_release release; // 🔓 GIL off (네이티브 CUDA 실행)

            const shim::Tensor* C = as_const_tensor_ptr(C_obj);
            shim::Tensor* gA      = as_tensor_ptr(gA_obj);
            shim::Tensor* gB      = as_tensor_ptr(gB_obj);
            shim::Tensor* gC      = as_tensor_ptr(gC_obj);
            shim::Tensor* gBias   = as_tensor_ptr(gBias_obj);
            shim::StreamHandle s  = as_stream(stream_obj);

            // ---- PerN 강제 세이프가드 ----
            const int64_t M = A.desc.shape.at(0);
            const int64_t N = B.desc.shape.at(1);
            if (gBias && gBias->data) {
                const auto& sh = gBias->desc.shape;
                const bool bad_perm =
                    (sh.size()==1 && sh[0]==M) ||
                    (sh.size()==2 && sh[0]==M && sh[1]==1);
                if (bad_perm) {
                    throw std::invalid_argument(
                        "[_ops_gemm::backward] gBias shape suggests PerM (len==M). "
                        "Bias grad for Dense must be PerN; allocate gBias as (1,N) or (N,).");
                }
            }

            NVTX_RANGE("gemm.backward", shim::nvtx::Color::Red);

            auto st = shim::GemmCudaBackward(
                A, B, C, gY, Z, gA, gB, gC, gBias, attrs, s
            );
            raise_if_not_ok(st, "backward");
        },
        "A"_a,
        "B"_a,
        "C"_a      = py::none(),
        "gY"_a,
        "Z"_a,
        "gA"_a     = py::none(),
        "gB"_a     = py::none(),
        "gC"_a     = py::none(),
        "gBias"_a  = py::none(),
        "attrs"_a,
        "stream"_a = py::none(),
        "Compute gradients for fused GEMM(+bias+activation) using pre-activation Z."
    );

    // =========================
    // 1-EX) 편의 오버로드: attrs 없이 호출
    // =========================
    m.def(
        "forward_ex",
        [](const shim::Tensor& A,
           const shim::Tensor& B,
           py::object bias_obj,
           shim::Tensor& Y,
           bool trans_a,
           bool trans_b,
           std::string act,
           bool with_bias,
           double leaky_slope,
           bool save_z,
           py::object Z_saved_obj,
           py::object stream_obj) {
            py::gil_scoped_release release; // 🔓 GIL off

            const shim::Tensor* Bias   = as_const_tensor_ptr(bias_obj);
            shim::Tensor*       Z_saved = as_tensor_ptr(Z_saved_obj);
            shim::StreamHandle  s       = as_stream(stream_obj);

            shim::GemmAttrs attrs{};
            attrs.trans_a     = trans_a;
            attrs.trans_b     = trans_b;
            attrs.act         = parse_act(act);
            attrs.with_bias   = with_bias;
            attrs.leaky_slope = static_cast<float>(leaky_slope);
            attrs.save_z      = save_z;

            if (attrs.with_bias && Bias == nullptr) {
                throw std::invalid_argument("with_bias=True but bias is None");
            }
            if (attrs.save_z) {
                if (!Z_saved || !Z_saved->data) {
                    throw std::invalid_argument("save_z=True requires Z_saved Tensor");
                }
            } else if (Z_saved && Z_saved->data) {
                attrs.save_z = true;
            }

            NVTX_RANGE("gemm.forward_ex", shim::nvtx::Color::Teal);

            auto st = shim::GemmCudaLaunch(A, B, Bias, Y, attrs, s, Z_saved);
            raise_if_not_ok(st, "forward_ex");
        },
        "A"_a,
        "B"_a,
        "bias"_a        = py::none(),
        "Y"_a,
        "trans_a"_a     = false,
        "trans_b"_a     = false,
        "act"_a         = "none",
        "with_bias"_a   = false,
        "leaky_slope"_a = 0.01,
        "save_z"_a      = false,
        "Z_saved"_a     = py::none(),
        "stream"_a      = py::none(),
        "Run fused GEMM with epilogue params provided as Python primitives. "
        "Optionally save pre-activation into Z_saved when save_z=True."
    );

    m.def(
        "backward_ex",
        [](const shim::Tensor& A,
           const shim::Tensor& B,
           py::object C_obj,
           const shim::Tensor& gY,
           const shim::Tensor& Z,
           py::object gA_obj,
           py::object gB_obj,
           py::object gC_obj,
           py::object gBias_obj,
           bool trans_a,
           bool trans_b,
           std::string act,
           bool with_bias,
           double leaky_slope,
           py::object stream_obj) {
            py::gil_scoped_release release; // 🔓 GIL off

            const shim::Tensor* C = as_const_tensor_ptr(C_obj);
            shim::Tensor* gA      = as_tensor_ptr(gA_obj);
            shim::Tensor* gB      = as_tensor_ptr(gB_obj);
            shim::Tensor* gC      = as_tensor_ptr(gC_obj);
            shim::Tensor* gBias   = as_tensor_ptr(gBias_obj);
            shim::StreamHandle s  = as_stream(stream_obj);

            shim::GemmAttrs attrs{};
            attrs.trans_a     = trans_a;
            attrs.trans_b     = trans_b;
            attrs.act         = parse_act(act);
            attrs.with_bias   = with_bias;
            attrs.leaky_slope = static_cast<float>(leaky_slope);

            // ---- PerN 강제 세이프가드 ----
            const int64_t M = A.desc.shape.at(0);
            const int64_t N = B.desc.shape.at(1);
            if (gBias && gBias->data) {
                const auto& sh = gBias->desc.shape;
                const bool bad_perm =
                    (sh.size()==1 && sh[0]==M) ||
                    (sh.size()==2 && sh[0]==M && sh[1]==1);
                if (bad_perm) {
                    throw std::invalid_argument(
                        "[_ops_gemm::backward_ex] gBias shape suggests PerM (len==M). "
                        "Bias grad for Dense must be PerN; allocate gBias as (1,N) or (N,).");
                }
            }

            NVTX_RANGE("gemm.backward_ex", shim::nvtx::Color::Magenta);

            auto st = shim::GemmCudaBackward(A, B, C, gY, Z, gA, gB, gC, gBias, attrs, s);
            raise_if_not_ok(st, "backward_ex");
        },
        "A"_a,
        "B"_a,
        "C"_a         = py::none(),
        "gY"_a,
        "Z"_a,
        "gA"_a        = py::none(),
        "gB"_a        = py::none(),
        "gC"_a        = py::none(),
        "gBias"_a     = py::none(),
        "trans_a"_a   = false,
        "trans_b"_a   = false,
        "act"_a       = "none",
        "with_bias"_a = false,
        "leaky_slope"_a = 0.01,
        "stream"_a    = py::none(),
        "Backward for fused GEMM with epilogue params provided as Python primitives."
    );

    // ===========================================
    // 3) Capture-safe backward_into (NO allocations)
    // ===========================================
    m.def(
        "backward_into",
        [](const shim::Tensor& A,
           const shim::Tensor& B,
           py::object C_obj,
           const shim::Tensor& gY,
           const shim::Tensor& Z,
           py::object gA_obj,
           py::object gB_obj,
           py::object gC_obj,
           py::object gBias_obj,
           const shim::GemmAttrs& attrs,
           py::object stream_obj,
           // --- workspaces ---
           uintptr_t dZ_ptr,
           uintptr_t lt_ws_ptr,
           size_t    lt_ws_bytes) {
            py::gil_scoped_release release; // 🔓 GIL off

            const shim::Tensor* C = as_const_tensor_ptr(C_obj);
            shim::Tensor* gA      = as_tensor_ptr(gA_obj);
            shim::Tensor* gB      = as_tensor_ptr(gB_obj);
            shim::Tensor* gC      = as_tensor_ptr(gC_obj);
            shim::Tensor* gBias   = as_tensor_ptr(gBias_obj);
            shim::StreamHandle s  = as_stream(stream_obj);

            const int64_t M = A.desc.shape.at(0);
            const int64_t N = B.desc.shape.at(1);

            if (gBias && gBias->data) {
                const auto& sh = gBias->desc.shape;
                const bool bad_perm =
                    (sh.size()==1 && sh[0]==M) ||
                    (sh.size()==2 && sh[0]==M && sh[1]==1);
                if (bad_perm) {
                    throw std::invalid_argument(
                        "[_ops_gemm::backward_into] gBias must be PerN (shape (1,N) or (N,))");
                }
            }

            if (!dZ_ptr) {
                throw std::invalid_argument(
                    "[_ops_gemm::backward_into] dZ_ptr is required for capture-safe path");
            }

            if ((lt_ws_ptr == 0) != (lt_ws_bytes == 0)) {
                throw std::invalid_argument(
                    "[_ops_gemm::backward_into] lt_workspace ptr/bytes must be both zero or both non-zero");
            }

            shim::GemmWorkspace ws{};
            ws.scratch            = reinterpret_cast<void*>(dZ_ptr);     // dZ buffer (ld=N 가정)
            ws.scratch_bytes      = static_cast<size_t>(M * N * sizeof(float));
            ws.lt_workspace       = reinterpret_cast<void*>(lt_ws_ptr);
            ws.lt_workspace_bytes = lt_ws_bytes;

            NVTX_RANGE("gemm.backward_into", shim::nvtx::Color::Yellow);

            auto st = shim::GemmCudaBackward(
                A, B, C, gY, Z, gA, gB, gC, gBias, attrs, s, &ws
            );
            raise_if_not_ok(st, "backward_into");
        },
        "A"_a, "B"_a,
        "C"_a      = py::none(),
        "gY"_a, "Z"_a,
        "gA"_a     = py::none(),
        "gB"_a     = py::none(),
        "gC"_a     = py::none(),
        "gBias"_a  = py::none(),
        "attrs"_a,
        "stream"_a = py::none(),
        "dZ_ptr"_a,
        "lt_ws_ptr"_a   = static_cast<uintptr_t>(0),
        "lt_ws_bytes"_a = static_cast<size_t>(0),
        "Capture-safe GEMM backward that uses preallocated workspaces (no malloc during capture)."
    );

    // === 별칭 (attrs 버전 유지용) ===
    m.def(
        "forward_ex_attrs",
        [](const shim::Tensor& A, const shim::Tensor& B, py::object bias_obj,
           shim::Tensor& Y, shim::GemmAttrs attrs, py::object Z_saved_obj, py::object stream_obj) {
            py::gil_scoped_release release;

            const shim::Tensor* Bias   = as_const_tensor_ptr(bias_obj);
            shim::Tensor*       Z_saved = as_tensor_ptr(Z_saved_obj);
            shim::StreamHandle  s       = as_stream(stream_obj);

            if (Z_saved && Z_saved->data && !attrs.save_z) attrs.save_z = true;
            if (attrs.save_z && (!Z_saved || !Z_saved->data))
                throw std::invalid_argument("[_ops_gemm::forward_ex_attrs] save_z=True requires Z_saved");

            auto st = shim::GemmCudaLaunch(A, B, Bias, Y, attrs, s, Z_saved);
            raise_if_not_ok(st, "forward_ex_attrs");
        },
        "A"_a, "B"_a, "bias"_a = py::none(), "Y"_a,
        "attrs"_a, "Z_saved"_a = py::none(), "stream"_a = py::none(),
        "Alias to `forward` that accepts GemmAttrs (kept for Python compatibility)."
    );

    m.def(
        "backward_ex_attrs",
        [](const shim::Tensor& A, const shim::Tensor& B, py::object C_obj,
           const shim::Tensor& gY, const shim::Tensor& Z,
           py::object gA_obj, py::object gB_obj, py::object gC_obj, py::object gBias_obj,
           const shim::GemmAttrs& attrs, py::object stream_obj) {
            py::gil_scoped_release release;

            const shim::Tensor* C = as_const_tensor_ptr(C_obj);
            shim::Tensor* gA      = as_tensor_ptr(gA_obj);
            shim::Tensor* gB      = as_tensor_ptr(gB_obj);
            shim::Tensor* gC      = as_tensor_ptr(gC_obj);
            shim::Tensor* gBias   = as_tensor_ptr(gBias_obj);
            shim::StreamHandle s  = as_stream(stream_obj);

            const int64_t M = A.desc.shape.at(0);
            const int64_t N = B.desc.shape.at(1);
            if (gBias && gBias->data) {
                const auto& sh = gBias->desc.shape;
                if ((sh.size()==1 && sh[0]==M) || (sh.size()==2 && sh[0]==M && sh[1]==1))
                    throw std::invalid_argument("[_ops_gemm::backward_ex_attrs] gBias must be PerN (1,N) or (N,)");
            }
            auto st = shim::GemmCudaBackward(A, B, C, gY, Z, gA, gB, gC, gBias, attrs, s);
            raise_if_not_ok(st, "backward_ex_attrs");
        },
        "A"_a, "B"_a, "C"_a = py::none(), "gY"_a, "Z"_a,
        "gA"_a = py::none(), "gB"_a = py::none(), "gC"_a = py::none(), "gBias"_a = py::none(),
        "attrs"_a, "stream"_a = py::none(),
        "Alias to `backward` that accepts GemmAttrs (kept for Python compatibility)."
    );

    // ===========================================
    // 4) Raw pointer 기반 API (Tensor/GemmAttrs 독립)
    // ===========================================
    m.def(
        "forward_raw",
        [](uintptr_t A_ptr,
           uintptr_t B_ptr,
           uintptr_t Bias_ptr,
           uintptr_t Y_ptr,
           int64_t M,
           int64_t K,
           int64_t N,
           bool trans_a,
           bool trans_b,
           std::string act,
           bool with_bias,
           double leaky_slope,
           bool save_z,
           uintptr_t Z_ptr,
           py::object stream_obj) {
            py::gil_scoped_release release;

            shim::StreamHandle s = as_stream(stream_obj);

            shim::Tensor A = make_tensor_2d_raw(A_ptr, M, K);
            shim::Tensor B = make_tensor_2d_raw(B_ptr, K, N);
            shim::Tensor Y = make_tensor_2d_raw(Y_ptr, M, N);

            shim::Tensor* Bias = nullptr;
            shim::Tensor  Bias_tensor{};
            if (with_bias) {
                if (!Bias_ptr) {
                    throw std::invalid_argument("[_ops_gemm::forward_raw] with_bias=True but Bias_ptr==0");
                }
                Bias_tensor = make_tensor_2d_raw(Bias_ptr, 1, N); // PerN: (1,N)
                Bias = &Bias_tensor;
            }

            shim::Tensor* Z_saved = nullptr;
            shim::Tensor  Z_tensor{};
            if (save_z) {
                if (!Z_ptr) {
                    throw std::invalid_argument("[_ops_gemm::forward_raw] save_z=True but Z_ptr==0");
                }
                Z_tensor = make_tensor_2d_raw(Z_ptr, M, N);
                Z_saved  = &Z_tensor;
            }

            shim::GemmAttrs attrs{};
            attrs.trans_a     = trans_a;
            attrs.trans_b     = trans_b;
            attrs.act         = parse_act(act);
            attrs.with_bias   = with_bias;
            attrs.leaky_slope = static_cast<float>(leaky_slope);
            attrs.save_z      = save_z;

            NVTX_RANGE("gemm.forward_raw", shim::nvtx::Color::Teal);

            auto st = shim::GemmCudaLaunch(A, B, Bias, Y, attrs, s, Z_saved);
            raise_if_not_ok(st, "forward_raw");
        },
        "A_ptr"_a,
        "B_ptr"_a,
        "Bias_ptr"_a,
        "Y_ptr"_a,
        "M"_a,
        "K"_a,
        "N"_a,
        "trans_a"_a     = false,
        "trans_b"_a     = false,
        "act"_a         = "none",
        "with_bias"_a   = false,
        "leaky_slope"_a = 0.01,
        "save_z"_a      = false,
        "Z_ptr"_a       = static_cast<uintptr_t>(0),
        "stream"_a      = py::none(),
        "Raw-pointer based fused GEMM(+bias+activation) for CUDA f32 row-major."
    );

    m.def(
        "backward_raw",
        [](uintptr_t A_ptr,
           uintptr_t B_ptr,
           uintptr_t C_ptr,
           uintptr_t gY_ptr,
           uintptr_t Z_ptr,
           uintptr_t gA_ptr,
           uintptr_t gB_ptr,
           uintptr_t gC_ptr,
           uintptr_t gBias_ptr,
           int64_t M,
           int64_t K,
           int64_t N,
           bool trans_a,
           bool trans_b,
           std::string act,
           bool with_bias,
           double leaky_slope,
           py::object stream_obj) {
            py::gil_scoped_release release;

            shim::StreamHandle s = as_stream(stream_obj);

            shim::Tensor A  = make_tensor_2d_raw(A_ptr,  M, K);
            shim::Tensor B  = make_tensor_2d_raw(B_ptr,  K, N);
            shim::Tensor gY = make_tensor_2d_raw(gY_ptr, M, N);
            shim::Tensor Z  = make_tensor_2d_raw(Z_ptr,  M, N);

            shim::Tensor* C = nullptr;
            shim::Tensor  C_tensor{};
            if (C_ptr) {
                C_tensor = make_tensor_2d_raw(C_ptr, M, N);
                C = &C_tensor;
            }

            shim::Tensor* gA = nullptr;
            shim::Tensor  gA_tensor{};
            if (gA_ptr) {
                gA_tensor = make_tensor_2d_raw(gA_ptr, M, K);
                gA = &gA_tensor;
            }

            shim::Tensor* gB = nullptr;
            shim::Tensor  gB_tensor{};
            if (gB_ptr) {
                gB_tensor = make_tensor_2d_raw(gB_ptr, K, N);
                gB = &gB_tensor;
            }

            shim::Tensor* gC = nullptr;
            shim::Tensor  gC_tensor{};
            if (gC_ptr) {
                gC_tensor = make_tensor_2d_raw(gC_ptr, M, N);
                gC = &gC_tensor;
            }

            shim::Tensor* gBias = nullptr;
            shim::Tensor  gBias_tensor{};
            if (gBias_ptr) {
                gBias_tensor = make_tensor_2d_raw(gBias_ptr, 1, N); // PerN
                gBias = &gBias_tensor;
            }

            shim::GemmAttrs attrs{};
            attrs.trans_a     = trans_a;
            attrs.trans_b     = trans_b;
            attrs.act         = parse_act(act);
            attrs.with_bias   = with_bias;
            attrs.leaky_slope = static_cast<float>(leaky_slope);
            attrs.save_z      = false; // bwd에서 save_z 의미 없음

            NVTX_RANGE("gemm.backward_raw", shim::nvtx::Color::Magenta);

            auto st = shim::GemmCudaBackward(
                A, B, C, gY, Z, gA, gB, gC, gBias, attrs, s
            );
            raise_if_not_ok(st, "backward_raw");
        },
        "A_ptr"_a,
        "B_ptr"_a,
        "C_ptr"_a      = static_cast<uintptr_t>(0),
        "gY_ptr"_a,
        "Z_ptr"_a,
        "gA_ptr"_a     = static_cast<uintptr_t>(0),
        "gB_ptr"_a     = static_cast<uintptr_t>(0),
        "gC_ptr"_a     = static_cast<uintptr_t>(0),
        "gBias_ptr"_a  = static_cast<uintptr_t>(0),
        "M"_a,
        "K"_a,
        "N"_a,
        "trans_a"_a    = false,
        "trans_b"_a    = false,
        "act"_a        = "none",
        "with_bias"_a  = false,
        "leaky_slope"_a= 0.01,
        "stream"_a     = py::none(),
        "Raw-pointer based backward for fused GEMM(+bias+activation)."
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
        "backward_into",
        "forward_ex_attrs", "backward_ex_attrs",
        // raw API
        "forward_raw", "backward_raw"
    );
}
