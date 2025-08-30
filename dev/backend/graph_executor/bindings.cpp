// bindings.cpp (revised, backward-compatible, enhanced)
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <memory>

#include "loss/loss_kernels.cuh"
#include "executor/run_graph.cuh"
#include "executor/run_graph_backward.cuh"
#include "executor/run_graph_with_loss.cuh"
#include "op_structs.cuh"
#include "optimizer/optimizer_types.cuh"
#include "optimizer/optimizer_config.cuh"
#include "optimizer/optimizer_kernels.cuh"

using namespace pybind11::literals;  // <-- _a 리터럴 활성화

namespace py = pybind11;

/* ------------------------------ Build Flags ------------------------------ */
#ifndef WEIGHT_DECAY_ENABLE
#define WEIGHT_DECAY_ENABLE 0
#endif
#ifndef AMSGRAD_ENABLE
#define AMSGRAD_ENABLE 0
#endif
#ifndef GLOBAL_NORM_CLIP_ENABLE
#define GLOBAL_NORM_CLIP_ENABLE 0
#endif
#ifndef GE_DEBUG_SYNC
// 1로 켜면 각 단계마다 cudaDeviceSynchronize()
#define GE_DEBUG_SYNC 0
#endif
#ifndef GE_VERBOSE
#define GE_VERBOSE 0
#endif
#ifndef GE_USE_NVTX
#define GE_USE_NVTX 0
#endif

// NVTX 사용시 -DGE_USE_NVTX=1 로 빌드
#if GE_USE_NVTX
#include <nvtx3/nvToolsExt.h>
#define NVTX_RANGE(name) nvtxRangePushA(name)
#define NVTX_POP()       nvtxRangePop()
#else
#define NVTX_RANGE(name) do{}while(0)
#define NVTX_POP()       do{}while(0)
#endif

// run_graph_backward가 반환한 gradient 버퍼를 바인딩에서 free할지 여부
#ifndef GE_FREE_GRADS
#define GE_FREE_GRADS 1
#endif

/* --------------------------------- CUDA ---------------------------------- */
#define CUDA_CHECK(stmt)                                                     \
    do {                                                                     \
        cudaError_t _e = (stmt);                                             \
        if (_e != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA error: ") +           \
                                     cudaGetErrorString(_e));                \
        }                                                                    \
    } while (0)

static inline OptimizerType to_opt(int v) {
    switch (v) {
        case 0: return OptimizerType::SGD;
        case 1: return OptimizerType::MOMENTUM;
        case 2: return OptimizerType::ADAM;
        default: return OptimizerType::ADAM;
    }
}

// 안전한 디바이스 포인터 캐스팅/검증
static inline float* as_device_ptr(uintptr_t p, const char* name){
    if (p == 0) throw std::runtime_error(std::string("null device ptr: ")+name);
    cudaPointerAttributes attr{};
    cudaError_t st = cudaPointerGetAttributes(&attr, reinterpret_cast<void*>(p));
    if (st != cudaSuccess) {
        throw std::runtime_error(std::string("bad device ptr (cudaPointerGetAttributes failed): ")+name);
    }
#if CUDART_VERSION >= 10000
    if (attr.type != cudaMemoryTypeDevice && attr.type != cudaMemoryTypeManaged) {
        throw std::runtime_error(std::string("not a device/managed ptr: ")+name);
    }
#else
    if (attr.memoryType != cudaMemoryTypeDevice && attr.memoryType != cudaMemoryTypeManaged) {
        throw std::runtime_error(std::string("not a device/managed ptr: ")+name);
    }
#endif
    return reinterpret_cast<float*>(p);
}
/* ------------------------ helpers: legacy -> vector ------------------------ */
static inline void normalize_legacy(OpStruct& n) {
    if (!n.input_id.empty() && n.inputs.empty())  n.inputs.push_back(n.input_id);
    if (!n.param_id.empty() && n.params.empty())  n.params.push_back(n.param_id);
}

// 값 복사 1회로 끝내고, 내부에서 in-place 정규화
static inline std::vector<OpStruct> normalize_graph(std::vector<OpStruct> E) {
    for (auto& n : E) normalize_legacy(n);
    return E;
}

static inline std::string loss_pred_input_id(const std::vector<OpStruct>& E) {
    if (E.empty()) return "";
    const OpStruct& last = E.back();
    if (last.op_type != OpType::LOSS) return "";
    if (!last.inputs.empty()) return last.inputs[0];
    return last.input_id;
}

// 디버그 친화 메시지
static inline void mismatch_throw(const char* tag, long long a, long long b) {
    throw std::runtime_error(std::string(tag) + " mismatch: " + std::to_string(a) + " vs " + std::to_string(b));
}

/* =============================== Entrypoints =============================== */

// Forward-only (out_host: None 허용, device_id: 옵션)
void run_graph_forward_entry(
    std::vector<OpStruct> E_in,
    const std::unordered_map<std::string, uintptr_t>& tensor_ptrs,
    std::unordered_map<std::string, Shape>& shapes,
    py::object out_host,                         // None 또는 numpy.ndarray(float32)
    const std::string& final_output_id,
    int batch_size,
    int device_id = -1                           // 음수면 변경 안 함
){
    if (E_in.empty()) throw std::runtime_error("empty graph");
    if (device_id >= 0) CUDA_CHECK(cudaSetDevice(device_id));

    const auto E = normalize_graph(std::move(E_in));

    std::unordered_map<std::string, float*> tensors;
    tensors.reserve(tensor_ptrs.size());
    for (const auto& kv : tensor_ptrs) {
        tensors[kv.first] = as_device_ptr(kv.second, kv.first.c_str());
    }

    // out_host가 None이면 디바이스 내에서만 유지
    float* out_ptr = nullptr;
    std::unique_ptr<py::array_t<float>> holder;
    if (!out_host.is_none()) {
        auto arr = out_host.cast<py::array_t<float>>();
        out_ptr = const_cast<float*>(arr.data());
        holder = std::make_unique<py::array_t<float>>(arr); // 라이프타임 보장
    }

    {
        py::gil_scoped_release nogil;
        NVTX_RANGE("forward");
        run_graph_cuda(E, tensors, shapes, /*out_host=*/out_ptr, final_output_id, batch_size);
        CUDA_CHECK(cudaGetLastError());
#if GE_DEBUG_SYNC
        CUDA_CHECK(cudaDeviceSynchronize());
#endif
        NVTX_POP();
    }
}

// Forward + Loss (device_id 옵션)
float run_graph_with_loss_entry(
    std::vector<OpStruct> E_in,
    const std::unordered_map<std::string, uintptr_t>& tensor_ptrs,
    std::unordered_map<std::string, Shape>& shapes,
    const std::string& final_output_id,
    const std::string& label_tensor_id,
    const std::string& loss_type,
    int batch_size,
    int device_id = -1
){
    if (E_in.empty()) throw std::runtime_error("empty graph");
    if (device_id >= 0) CUDA_CHECK(cudaSetDevice(device_id));

    const auto E = normalize_graph(std::move(E_in));

    std::unordered_map<std::string, float*> tensors;
    tensors.reserve(tensor_ptrs.size());
    for (const auto& kv : tensor_ptrs) {
        tensors[kv.first] = as_device_ptr(kv.second, kv.first.c_str());
    }

    float loss = 0.f;
    {
        py::gil_scoped_release nogil;

        NVTX_RANGE("forward(warm)+loss");

        // LOSS가 마지막이면 예측 텐서 id 추적
        std::string pred_id = final_output_id;
        if (!E.empty() && E.back().op_type == OpType::LOSS) {
            const auto cand = loss_pred_input_id(E);
            if (!cand.empty()) pred_id = cand;
        }

        // forward (host copy 없음)
        run_graph_cuda(E, tensors, shapes, /*out_host=*/nullptr, pred_id, batch_size);
        CUDA_CHECK(cudaGetLastError());

        // ---- Loss ----
        auto it_pred = tensors.find(pred_id);
        auto it_true = tensors.find(label_tensor_id);
        auto it_pshp = shapes.find(pred_id);
        auto it_tshp = shapes.find(label_tensor_id);
        if (it_pred == tensors.end() || it_true == tensors.end() ||
            it_pshp == shapes.end()  || it_tshp == shapes.end())
            throw std::runtime_error("loss: missing y_pred/y_true/shape");

        float* y_pred = it_pred->second;
        float* y_true = it_true->second;
        const Shape sp = it_pshp->second;
        const Shape st = it_tshp->second;

        const int rows_per_sample = sp.rows; // 시퀀스면 >1
        const int C = sp.cols;
        const int B = batch_size * rows_per_sample;
        const long long N = 1LL * B * C;

        if (loss_type == "mse") {
            const long long trueN = 1LL * st.rows * st.cols * batch_size;
            if (trueN != N) mismatch_throw("loss(mse) total elements", trueN, N);
            loss = compute_mse_loss_cuda(y_true, y_pred, (int)N);
        } else if (loss_type == "binary_crossentropy" || loss_type == "bce") {
            const long long trueN = 1LL * st.rows * st.cols * batch_size;
            if (trueN != N) mismatch_throw("loss(bce) total elements", trueN, N);
            loss = compute_bce_loss_cuda(y_true, y_pred, (int)N);
        } else if (loss_type == "cce") {
            if (st.rows != rows_per_sample || st.cols != C) {
                throw std::runtime_error("loss(cce): y_true per-sample shape mismatch");
            }
            loss = compute_cce_loss_cuda(y_true, y_pred, B, C);
        } else {
            throw std::runtime_error("loss: unsupported type");
        }
        CUDA_CHECK(cudaGetLastError());
#if GE_DEBUG_SYNC
        CUDA_CHECK(cudaDeviceSynchronize());
#endif
        NVTX_POP();
    }
    return loss;
}

// Backward (device_id 옵션, gradients 포인터 딕셔너리 반환)
py::dict run_graph_backward_entry(
    std::vector<OpStruct> E_in,
    const std::unordered_map<std::string, uintptr_t>& tensor_ptrs,
    std::unordered_map<std::string, Shape>& shapes,
    const std::unordered_map<std::string, uintptr_t>& /*gradient_ptrs_unused*/,
    const std::string& final_output_id,
    int batch_size,
    int device_id = -1
){
    if (E_in.empty()) throw std::runtime_error("empty graph");
    if (device_id >= 0) CUDA_CHECK(cudaSetDevice(device_id));

    const auto E = normalize_graph(std::move(E_in));

    std::unordered_map<std::string, float*> tensors;
    tensors.reserve(tensor_ptrs.size());
    for (const auto& kv : tensor_ptrs) {
        tensors[kv.first] = as_device_ptr(kv.second, kv.first.c_str());
    }

    // LOSS의 입력(y_pred) id
    std::string pred_id = final_output_id;
    if (!E.empty() && E.back().op_type == OpType::LOSS) {
        const auto cand = loss_pred_input_id(E);
        if (!cand.empty()) pred_id = cand;
    }

    std::unordered_map<std::string, float*> gradients;
    {
        py::gil_scoped_release nogil;

        NVTX_RANGE("forward(warm)");
        run_graph_cuda(E, tensors, shapes, /*out_host=*/nullptr, pred_id, batch_size);
        CUDA_CHECK(cudaGetLastError());
#if GE_DEBUG_SYNC
        CUDA_CHECK(cudaDeviceSynchronize());
#endif
        NVTX_POP();

        NVTX_RANGE("backward");
        run_graph_backward(E, tensors, shapes, gradients, final_output_id, batch_size);
        CUDA_CHECK(cudaGetLastError());
#if GE_DEBUG_SYNC
        CUDA_CHECK(cudaDeviceSynchronize());
#endif
        NVTX_POP();
    }

    py::dict result;
    for (const auto& kv : gradients)
        result[py::str(kv.first)] = reinterpret_cast<uintptr_t>(kv.second);
    return result;
}

// Train step: Fwd(warm)+Loss -> Bwd -> Optimizer (device_id 옵션)
float train_step_entry(
    std::vector<OpStruct> E_in,
    const std::unordered_map<std::string, uintptr_t>& tensor_ptrs,
    std::unordered_map<std::string, Shape>& shapes,
    const std::string& final_output_id,
    const std::string& label_tensor_id,
    const std::string& loss_type,
    int batch_size,
    OptimizerType opt_type,
    float lr = 0.01f,
    float beta1 = 0.9f,
    float beta2 = 0.999f,
    float eps = 1e-8f,
    int timestep = 1,
#if WEIGHT_DECAY_ENABLE
    float weight_decay = 0.0f,
#endif
    const std::unordered_map<std::string, uintptr_t>& velocity_ptrs = {},
    const std::unordered_map<std::string, uintptr_t>& m_ptrs = {},
    const std::unordered_map<std::string, uintptr_t>& v_ptrs = {}
#if AMSGRAD_ENABLE
   ,const std::unordered_map<std::string, uintptr_t>& vhat_max_ptrs = {}
#endif
   ,int device_id = -1
){
    if (E_in.empty()) throw std::runtime_error("empty graph");
    if (device_id >= 0) CUDA_CHECK(cudaSetDevice(device_id));

    const auto E = normalize_graph(std::move(E_in));

    std::unordered_map<std::string, float*> tensors;
    tensors.reserve(tensor_ptrs.size());
    for (const auto& kv : tensor_ptrs) {
        tensors[kv.first] = as_device_ptr(kv.second, kv.first.c_str());
    }

    float loss = 0.f;
    std::unordered_map<std::string, float*> gradients;

    {
        py::gil_scoped_release nogil;

        // 예측 텐서 id 결정
        std::string pred_id = final_output_id;
        if (!E.empty() && E.back().op_type == OpType::LOSS) {
            const auto cand = loss_pred_input_id(E);
            if (!cand.empty()) pred_id = cand;
        }

        NVTX_RANGE("forward(warm)");
        run_graph_cuda(E, tensors, shapes, /*out_host=*/nullptr, pred_id, batch_size);
        CUDA_CHECK(cudaGetLastError());
#if GE_DEBUG_SYNC
        CUDA_CHECK(cudaDeviceSynchronize());
#endif
        NVTX_POP();

        // ---- Loss ----
        NVTX_RANGE("loss");
        auto it_pred = tensors.find(pred_id);
        auto it_true = tensors.find(label_tensor_id);
        auto it_pshp = shapes.find(pred_id);
        auto it_tshp = shapes.find(label_tensor_id);
        if (it_pred == tensors.end() || it_true == tensors.end() ||
            it_pshp == shapes.end()  || it_tshp == shapes.end())
            throw std::runtime_error("loss: missing y_pred/y_true/shape");

        float* y_pred = it_pred->second;
        float* y_true = it_true->second;
        const Shape sp = it_pshp->second;
        const Shape st = it_tshp->second;

        const int rows_per_sample = sp.rows;
        const int C = sp.cols;
        const int B = batch_size * rows_per_sample;
        const long long N = 1LL * B * C;

        if (loss_type == "mse") {
            const long long trueN = 1LL * st.rows * st.cols * batch_size;
            if (trueN != N) mismatch_throw("loss(mse) total elements", trueN, N);
            loss = compute_mse_loss_cuda(y_true, y_pred, (int)N);
        } else if (loss_type == "binary_crossentropy" || loss_type == "bce") {
            const long long trueN = 1LL * st.rows * st.cols * batch_size;
            if (trueN != N) mismatch_throw("loss(bce) total elements", trueN, N);
            loss = compute_bce_loss_cuda(y_true, y_pred, (int)N);
        } else if (loss_type == "cce") {
            if (st.rows != rows_per_sample || st.cols != C) {
                throw std::runtime_error("loss(cce): y_true per-sample shape mismatch");
            }
            loss = compute_cce_loss_cuda(y_true, y_pred, B, C);
        } else {
            throw std::runtime_error("loss: unsupported type");
        }
        CUDA_CHECK(cudaGetLastError());
#if GE_DEBUG_SYNC
        CUDA_CHECK(cudaDeviceSynchronize());
#endif
        NVTX_POP();

        // ---- Backward ----
        NVTX_RANGE("backward");
        run_graph_backward(E, tensors, shapes, gradients, final_output_id, batch_size);
        CUDA_CHECK(cudaGetLastError());
#if GE_DEBUG_SYNC
        CUDA_CHECK(cudaDeviceSynchronize());
#endif
        NVTX_POP();

        // ---- Optimizer ----
        NVTX_RANGE("optimizer");
        std::set<std::string> trainable_params;
        for (const auto& op : E) {
            if (!op.params.empty()) {
                trainable_params.insert(op.params.begin(), op.params.end());
            } else if (!op.param_id.empty()) {
                trainable_params.insert(op.param_id);
            }
        }

        for (const auto& name : trainable_params) {
            auto t_it = tensors.find(name);
            auto g_it = gradients.find(name);
            auto s_it = shapes.find(name);
            if (t_it == tensors.end() || g_it == gradients.end() || s_it == shapes.end()) {
#if GE_VERBOSE
                std::cerr << "[warn] missing tensor/grad/shape for param " << name << "\n";
#endif
                continue;
            }

            float*       param_ptr = t_it->second;
            const float* grad_ptr  = g_it->second;
            const Shape& shp       = s_it->second;
            const int size = shp.rows * shp.cols;

            uintptr_t vel_u = 0, m_u = 0, v_u = 0;
#if AMSGRAD_ENABLE
            uintptr_t vhat_u = 0;
#endif
            if (velocity_ptrs.count(name)) vel_u = velocity_ptrs.at(name);
            if (m_ptrs.count(name))        m_u   = m_ptrs.at(name);
            if (v_ptrs.count(name))        v_u   = v_ptrs.at(name);
#if AMSGRAD_ENABLE
            if (vhat_max_ptrs.count(name)) vhat_u = vhat_max_ptrs.at(name);
#endif

            optimizer_update_cuda(
                param_ptr, grad_ptr,
                reinterpret_cast<float*>(vel_u),
                reinterpret_cast<float*>(m_u),
                reinterpret_cast<float*>(v_u),
#if AMSGRAD_ENABLE
                reinterpret_cast<float*>(vhat_u),
#endif
                lr, beta1, beta2, eps,
#if WEIGHT_DECAY_ENABLE
                weight_decay,
#endif
                size, opt_type, timestep
            );
            CUDA_CHECK(cudaGetLastError());
#if GE_DEBUG_SYNC
            CUDA_CHECK(cudaDeviceSynchronize());
#endif
        }

#if GE_FREE_GRADS
        // free grad buffers (중복 포인터 dedup)
        std::unordered_set<const float*> freed;
        for (const auto& kv : gradients) {
            const float* p = kv.second;
            if (p && freed.insert(p).second) CUDA_CHECK(cudaFree(const_cast<float*>(p)));
        }
#endif
        NVTX_POP();
    }

    return loss;
}

/* ============================ Pinned Host Utils ============================ */
static py::array pinned_float_array_1d(py::ssize_t n) {
    void* p = nullptr;
    CUDA_CHECK(cudaHostAlloc(&p, n * sizeof(float), cudaHostAllocPortable));
    auto capsule = py::capsule(p, [](void* ptr){ cudaFreeHost(ptr); });

    // buffer_info 생성 시에도 py::ssize_t 사용
    return py::array(py::buffer_info(
        p,
        sizeof(float),
        py::format_descriptor<float>::format(),
        1,
        { n },                              // std::initializer_list<py::ssize_t>
        { static_cast<py::ssize_t>(sizeof(float)) }
    ), capsule);
}

/* =============================== Pybind Module ============================= */

PYBIND11_MODULE(graph_executor, m) {
    // 빌드 옵션 노출 (디버깅 편의)
    {
        py::dict flags;
        // 매크로가 정수형이 아닐 수도 있으니 안전하게 (int) 캐스팅
        flags["WEIGHT_DECAY_ENABLE"]      = (int)WEIGHT_DECAY_ENABLE;
        flags["AMSGRAD_ENABLE"]           = (int)AMSGRAD_ENABLE;
        flags["GLOBAL_NORM_CLIP_ENABLE"]  = (int)GLOBAL_NORM_CLIP_ENABLE;
        flags["GE_DEBUG_SYNC"]            = (int)GE_DEBUG_SYNC;
        flags["GE_USE_NVTX"]              = (int)GE_USE_NVTX;

        m.attr("_build_flags") = std::move(flags);
    }
    // Enums
    py::enum_<OpType>(m, "OpType")
        .value("MATMUL", OpType::MATMUL)
        .value("ADD", OpType::ADD)
        .value("RELU", OpType::RELU)
        .value("SIGMOID", OpType::SIGMOID)
        .value("TANH", OpType::TANH)
        .value("FLATTEN", OpType::FLATTEN)
        .value("CONV2D", OpType::CONV2D)
        .value("LOSS", OpType::LOSS)
        .value("LEAKY_RELU", OpType::LEAKY_RELU)
        .value("ELU", OpType::ELU)
        .value("GELU", OpType::GELU)
        .value("SILU", OpType::SILU)
        .value("SOFTMAX", OpType::SOFTMAX)
        .value("POOL_MAX", OpType::POOL_MAX)
        .value("POOL_AVG", OpType::POOL_AVG)
        .value("ADD_BIAS", OpType::ADD_BIAS)
        .value("SLICE_TIME", OpType::SLICE_TIME)
        .value("CONCAT_TIME", OpType::CONCAT_TIME)
        .value("FILL_ZERO", OpType::FILL_ZERO)
        .value("RNN", OpType::RNN)
        .export_values();

    py::enum_<OptimizerType>(m, "OptimizerType")
        .value("SGD", OptimizerType::SGD)
        .value("MOMENTUM", OptimizerType::MOMENTUM)
        .value("ADAM", OptimizerType::ADAM)
        .export_values();

    py::class_<OpExtraParams>(m, "OpExtraParams")
        .def(py::init([](){
            OpExtraParams e{};
            e.alpha = 0.01f; e.gelu_tanh = 1; e.temperature = 1.0f; e.axis = 1;
            e.dilation_h = 1; e.dilation_w = 1; e.count_include_pad = false;
            return e;
        }))
        .def_readwrite("kernel_h", &OpExtraParams::kernel_h)
        .def_readwrite("kernel_w", &OpExtraParams::kernel_w)
        .def_readwrite("stride_h", &OpExtraParams::stride_h)
        .def_readwrite("stride_w", &OpExtraParams::stride_w)
        .def_readwrite("padding_h", &OpExtraParams::padding_h)
        .def_readwrite("padding_w", &OpExtraParams::padding_w)
        .def_readwrite("dilation_h", &OpExtraParams::dilation_h)
        .def_readwrite("dilation_w", &OpExtraParams::dilation_w)
        .def_readwrite("count_include_pad", &OpExtraParams::count_include_pad)
        .def_readwrite("input_h", &OpExtraParams::input_h)
        .def_readwrite("input_w", &OpExtraParams::input_w)
        .def_readwrite("input_c", &OpExtraParams::input_c)
        .def_readwrite("output_c", &OpExtraParams::output_c)
        .def_readwrite("batch_size", &OpExtraParams::batch_size)
        .def_readwrite("time_steps", &OpExtraParams::time_steps)
        .def_readwrite("hidden_size", &OpExtraParams::hidden_size)
        .def_readwrite("num_layers", &OpExtraParams::num_layers)
        .def_readwrite("use_bias", &OpExtraParams::use_bias)
        .def_readwrite("label_id", &OpExtraParams::label_id)
        .def_readwrite("loss_type", &OpExtraParams::loss_type)
        .def_readwrite("alpha", &OpExtraParams::alpha)
        .def_readwrite("gelu_tanh", &OpExtraParams::gelu_tanh)
        .def_readwrite("temperature", &OpExtraParams::temperature)
        .def_readwrite("axis", &OpExtraParams::axis)
        .def_readwrite("time_index", &OpExtraParams::time_index)
        .def_readwrite("concat_count", &OpExtraParams::concat_count);

    py::class_<Shape>(m, "Shape")
        .def(py::init<int, int>())
        .def_readwrite("rows", &Shape::rows)
        .def_readwrite("cols", &Shape::cols);

    // OpStruct: legacy + vector API
    py::class_<OpStruct>(m, "OpStruct")
        .def(py::init<>())
        // legacy ctor
        .def(py::init<OpType, std::string, std::string, std::string, OpExtraParams>(),
             py::arg("op_type"), py::arg("input_id"), py::arg("param_id"),
             py::arg("output_id"), py::arg("extra_params") = OpExtraParams())
        // vector ctor
        .def(py::init<OpType, std::vector<std::string>, std::vector<std::string>, std::string, OpExtraParams>(),
             py::arg("op_type"), py::arg("inputs"), py::arg("params"),
             py::arg("output_id"), py::arg("extra_params") = OpExtraParams())
        .def_readwrite("op_type", &OpStruct::op_type)
        .def_readwrite("input_id", &OpStruct::input_id)
        .def_readwrite("param_id", &OpStruct::param_id)
        .def_readwrite("inputs",   &OpStruct::inputs)
        .def_readwrite("params",   &OpStruct::params)
        .def_readwrite("output_id",&OpStruct::output_id)
        .def_readwrite("extra_params", &OpStruct::extra_params);

    /* ------------------------------- Graph APIs ------------------------------ */
    // Forward (out_host: None 허용, device_id 옵션)
    m.def("run_graph_forward_entry", &run_graph_forward_entry,
        py::arg("E"),
        py::arg("tensors"),
        py::arg("shapes"),
        py::arg("out_host"),                   // None 또는 numpy(float32)
        py::arg("final_output_id"),
        py::arg("batch_size"),
        py::arg("device_id") = -1
    );

    // Forward + Loss
    m.def("run_graph_with_loss_entry", &run_graph_with_loss_entry,
        py::arg("E"),
        py::arg("tensors"),
        py::arg("shapes"),
        py::arg("final_output_id"),
        py::arg("label_tensor_id"),
        py::arg("loss_type"),
        py::arg("batch_size"),
        py::arg("device_id") = -1
    );

    // Backward
    m.def("run_graph_backward_entry", &run_graph_backward_entry,
        py::arg("E"),
        py::arg("tensors"),
        py::arg("shapes"),
        py::arg("gradients"),
        py::arg("final_output_id"),
        py::arg("batch_size"),
        py::arg("device_id") = -1
    );

    // Optimizer: enum version
    m.def("optimizer_update",
        [](uintptr_t param_ptr, uintptr_t grad_ptr,
           uintptr_t velocity_ptr, uintptr_t m_ptr, uintptr_t v_ptr,
#if WEIGHT_DECAY_ENABLE
           float weight_decay,
#endif
           float lr, float beta1, float beta2, float eps,
           int size, OptimizerType opt_type, int timestep
#if AMSGRAD_ENABLE
          ,uintptr_t vhat_max_ptr
#endif
        ){
            py::gil_scoped_release nogil;
            optimizer_update_cuda(
                as_device_ptr(param_ptr, "param"),
                reinterpret_cast<const float*>(as_device_ptr(grad_ptr, "grad")),
                velocity_ptr ? as_device_ptr(velocity_ptr, "velocity") : nullptr,
                m_ptr ? as_device_ptr(m_ptr, "m") : nullptr,
                v_ptr ? as_device_ptr(v_ptr, "v") : nullptr,
#if AMSGRAD_ENABLE
                vhat_max_ptr ? as_device_ptr(vhat_max_ptr, "vhat_max") : nullptr,
#endif
                lr, beta1, beta2, eps,
#if WEIGHT_DECAY_ENABLE
                weight_decay,
#endif
                size, opt_type, timestep
            );
            CUDA_CHECK(cudaGetLastError());
#if GE_DEBUG_SYNC
            CUDA_CHECK(cudaDeviceSynchronize());
#endif
        },
        py::arg("param_ptr"), py::arg("grad_ptr"),
        py::arg("velocity_ptr") = 0, py::arg("m_ptr") = 0, py::arg("v_ptr") = 0,
#if WEIGHT_DECAY_ENABLE
        py::arg("weight_decay") = 0.0f,
#endif
        py::arg("lr") = 0.01f, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f,
        py::arg("size"), py::arg("opt_type"), py::arg("timestep") = 1
#if AMSGRAD_ENABLE
       ,py::arg("vhat_max_ptr") = 0
#endif
    );

    // Optimizer: legacy int version
    m.def("optimizer_update",
        [](uintptr_t param_ptr, uintptr_t grad_ptr,
           uintptr_t velocity_ptr, uintptr_t m_ptr, uintptr_t v_ptr,
#if WEIGHT_DECAY_ENABLE
           float weight_decay,
#endif
           float lr, float beta1, float beta2, float eps,
           int size, int opt_type_int, int timestep
#if AMSGRAD_ENABLE
          ,uintptr_t vhat_max_ptr
#endif
        ){
            const OptimizerType opt_type = to_opt(opt_type_int);
            py::gil_scoped_release nogil;
            optimizer_update_cuda(
                as_device_ptr(param_ptr, "param"),
                reinterpret_cast<const float*>(as_device_ptr(grad_ptr, "grad")),
                velocity_ptr ? as_device_ptr(velocity_ptr, "velocity") : nullptr,
                m_ptr ? as_device_ptr(m_ptr, "m") : nullptr,
                v_ptr ? as_device_ptr(v_ptr, "v") : nullptr,
#if AMSGRAD_ENABLE
                vhat_max_ptr ? as_device_ptr(vhat_max_ptr, "vhat_max") : nullptr,
#endif
                lr, beta1, beta2, eps,
#if WEIGHT_DECAY_ENABLE
                weight_decay,
#endif
                size, opt_type, timestep
            );
            CUDA_CHECK(cudaGetLastError());
#if GE_DEBUG_SYNC
            CUDA_CHECK(cudaDeviceSynchronize());
#endif
        },
        py::arg("param_ptr"), py::arg("grad_ptr"),
        py::arg("velocity_ptr") = 0, py::arg("m_ptr") = 0, py::arg("v_ptr") = 0,
#if WEIGHT_DECAY_ENABLE
        py::arg("weight_decay") = 0.0f,
#endif
        py::arg("lr") = 0.01f, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f,
        py::arg("size"), py::arg("opt_type"), py::arg("timestep") = 1
#if AMSGRAD_ENABLE
       ,py::arg("vhat_max_ptr") = 0
#endif
    );

    // One-shot train step
    m.def("train_step_entry", &train_step_entry,
        py::arg("E"),
        py::arg("tensors"),
        py::arg("shapes"),
        py::arg("final_output_id"),
        py::arg("label_tensor_id"),
        py::arg("loss_type"),
        py::arg("batch_size"),
        py::arg("opt_type"),
        py::arg("lr") = 0.01f,
        py::arg("beta1") = 0.9f,
        py::arg("beta2") = 0.999f,
        py::arg("eps") = 1e-8f,
        py::arg("timestep") = 1,
#if WEIGHT_DECAY_ENABLE
        py::arg("weight_decay") = 0.0f,
#endif
        py::arg("velocity_ptrs") = std::unordered_map<std::string, uintptr_t>{},
        py::arg("m_ptrs") = std::unordered_map<std::string, uintptr_t>{},
        py::arg("v_ptrs") = std::unordered_map<std::string, uintptr_t>{}
#if AMSGRAD_ENABLE
       ,py::arg("vhat_max_ptrs") = std::unordered_map<std::string, uintptr_t>{}
#endif
       ,py::arg("device_id") = -1
    );

    /* --------------------------- Pinned Host Utils -------------------------- */
    m.def("pinned_float_array_1d", &pinned_float_array_1d,
          py::arg("n"),
          "Allocate 1D float32 NumPy array in pinned host memory.");
}
