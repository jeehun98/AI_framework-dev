// bindings.cpp (revised)

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
#include <stdexcept>  // added

#include "loss_kernels.cuh"
#include "run_graph.cuh"
#include "run_graph_backward.cuh"
#include "run_graph_with_loss.cuh"
#include "op_structs.cuh"
#include "optimizer_types.cuh"

#ifndef WEIGHT_DECAY_ENABLE
#define WEIGHT_DECAY_ENABLE 0
#endif
#ifndef AMSGRAD_ENABLE
#define AMSGRAD_ENABLE 0
#endif
#ifndef GLOBAL_NORM_CLIP_ENABLE
#define GLOBAL_NORM_CLIP_ENABLE 0
#endif
// Optional debug knobs:
// #define GE_DEBUG_SYNC 1
// #define GE_VERBOSE 1

#include "optimizer_config.cuh"
#include "optimizer_kernels.cuh"

namespace py = pybind11;

// ---- CUDA error guard ----
#define CUDA_CHECK(stmt)                                                     \
    do {                                                                     \
        cudaError_t _e = (stmt);                                             \
        if (_e != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA error: ") +           \
                                     cudaGetErrorString(_e));                \
        }                                                                    \
    } while (0)

// ---- safe cast ----
static inline OptimizerType to_opt(int v) {
    switch (v) {
        case 0: return OptimizerType::SGD;
        case 1: return OptimizerType::MOMENTUM;
        case 2: return OptimizerType::ADAM;
        default: return OptimizerType::ADAM;
    }
}

// ===================== Entrypoints =====================

// Forward-only
void run_graph_forward_entry(
    const std::vector<OpStruct>& E,
    const std::unordered_map<std::string, uintptr_t>& tensor_ptrs,
    std::unordered_map<std::string, Shape>& shapes,
    py::array_t<float> out_host,
    const std::string& final_output_id,
    int batch_size)
{
    if (E.empty()) throw std::runtime_error("empty graph");

    std::unordered_map<std::string, float*> tensors;
    tensors.reserve(tensor_ptrs.size());
    for (const auto& kv : tensor_ptrs)
        tensors[kv.first] = reinterpret_cast<float*>(kv.second);

    float* out_ptr = out_host.mutable_data();
    {
        py::gil_scoped_release nogil;
        run_graph_cuda(E, tensors, shapes, out_ptr, final_output_id, batch_size);
        CUDA_CHECK(cudaGetLastError());
#ifdef GE_DEBUG_SYNC
        CUDA_CHECK(cudaDeviceSynchronize());
#endif
    }
}

// Forward + Loss
float run_graph_with_loss_entry(
    const std::vector<OpStruct>& E,
    const std::unordered_map<std::string, uintptr_t>& tensor_ptrs,
    std::unordered_map<std::string, Shape>& shapes,
    const std::string& final_output_id,
    const std::string& label_tensor_id,
    const std::string& loss_type,
    int batch_size)
{
    if (E.empty()) throw std::runtime_error("empty graph");

    std::unordered_map<std::string, float*> tensors;
    tensors.reserve(tensor_ptrs.size());
    for (const auto& kv : tensor_ptrs)
        tensors[kv.first] = reinterpret_cast<float*>(kv.second);

    float loss = 0.f;
    {
        py::gil_scoped_release nogil;
        loss = run_graph_with_loss_cuda(E, tensors, shapes,
                                        final_output_id, label_tensor_id,
                                        loss_type, batch_size);
        CUDA_CHECK(cudaGetLastError());
#ifdef GE_DEBUG_SYNC
        CUDA_CHECK(cudaDeviceSynchronize());
#endif
    }
    return loss;
}

// Backward
py::dict run_graph_backward_entry(
    const std::vector<OpStruct>& E,
    const std::unordered_map<std::string, uintptr_t>& tensor_ptrs,
    std::unordered_map<std::string, Shape>& shapes,
    const std::unordered_map<std::string, uintptr_t>& /*gradient_ptrs_unused*/,
    const std::string& final_output_id,
    int batch_size)
{
    if (E.empty()) throw std::runtime_error("empty graph");

    std::unordered_map<std::string, float*> tensors;
    tensors.reserve(tensor_ptrs.size());
    for (const auto& kv : tensor_ptrs)
        tensors[kv.first] = reinterpret_cast<float*>(kv.second);

    // If final node is LOSS, we also need its input (y_pred) to exist on device
    std::string pred_id = final_output_id;
    if (!E.empty() && E.back().op_type == OpType::LOSS)
        pred_id = E.back().input_id;

    std::unordered_map<std::string, float*> gradients;
    {
        py::gil_scoped_release nogil;
        // Warm forward to produce y_pred on device only
        run_graph_cuda(E, tensors, shapes, /*out_host=*/nullptr, pred_id, batch_size);
        CUDA_CHECK(cudaGetLastError());
#ifdef GE_DEBUG_SYNC
        CUDA_CHECK(cudaDeviceSynchronize());
#endif

        run_graph_backward(E, tensors, shapes, gradients, final_output_id, batch_size);
        CUDA_CHECK(cudaGetLastError());
#ifdef GE_DEBUG_SYNC
        CUDA_CHECK(cudaDeviceSynchronize());
#endif
    }

    py::dict result;
    for (const auto& kv : gradients)
        result[py::str(kv.first)] = reinterpret_cast<uintptr_t>(kv.second);
    return result;
}

// Train step (Fwd+Loss -> Bwd -> Optimizer)
float train_step_entry(
    const std::vector<OpStruct>& E,
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
)
{
    if (E.empty()) throw std::runtime_error("empty graph");

    std::unordered_map<std::string, float*> tensors;
    tensors.reserve(tensor_ptrs.size());
    for (const auto& kv : tensor_ptrs)
        tensors[kv.first] = reinterpret_cast<float*>(kv.second);

    float loss = 0.f;
    std::unordered_map<std::string, float*> gradients;

    {   // ---- Forward (keep intermediates) ----
        py::gil_scoped_release nogil;

        // final_output_id가 LOSS라면 그 입력이 y_pred
        std::string pred_id = final_output_id;
        if (!E.empty() && E.back().op_type == OpType::LOSS)
            pred_id = E.back().input_id;

        // Forward 실행. out_host=nullptr → 호스트 복사 없음. 중간 텐서 유지.
        run_graph_cuda(E, tensors, shapes, /*out_host=*/nullptr, pred_id, batch_size);
        CUDA_CHECK(cudaGetLastError());
#ifdef GE_DEBUG_SYNC
        CUDA_CHECK(cudaDeviceSynchronize());
#endif

        // ---- Loss 계산 ----
        auto it_pred = tensors.find(pred_id);
        auto it_true = tensors.find(label_tensor_id);
        auto it_pshp = shapes.find(pred_id);
        auto it_tshp = shapes.find(label_tensor_id);
        if (it_pred == tensors.end() || it_true == tensors.end() ||
            it_pshp == shapes.end()  || it_tshp == shapes.end())
            throw std::runtime_error("loss: missing y_pred/y_true/shape");

        float* y_pred = it_pred->second;
        float* y_true = it_true->second;
        const Shape sp = it_pshp->second; // per-sample
        const Shape st = it_tshp->second; // per-sample

        const int rows_per_sample = sp.rows; // may be 1 or seq_len
        const int C = sp.cols;
        const int B = batch_size * rows_per_sample;
        const int N = B * C;

        if (loss_type == "mse") {
            if (st.rows * st.cols * batch_size != N)
                throw std::runtime_error("loss(mse): y_true size mismatch");
            loss = compute_mse_loss_cuda(y_true, y_pred, N);
        } else if (loss_type == "binary_crossentropy" || loss_type == "bce") {
            if (st.rows * st.cols * batch_size != N)
                throw std::runtime_error("loss(bce): y_true size mismatch");
            loss = compute_bce_loss_cuda(y_true, y_pred, N);
        } else if (loss_type == "cce") {
            if (st.rows != rows_per_sample || st.cols != C)
                throw std::runtime_error("loss(cce): y_true per-sample shape mismatch");
            loss = compute_cce_loss_cuda(y_true, y_pred, B, C);
        } else {
            throw std::runtime_error("loss: unsupported type");
        }
        CUDA_CHECK(cudaGetLastError());
#ifdef GE_DEBUG_SYNC
        CUDA_CHECK(cudaDeviceSynchronize());
#endif

        // ---- Backward (중간 텐서 해제 금지) ----
        run_graph_backward(E, tensors, shapes, gradients, final_output_id, batch_size);
        CUDA_CHECK(cudaGetLastError());
#ifdef GE_DEBUG_SYNC
        CUDA_CHECK(cudaDeviceSynchronize());
#endif

        // ---- Optimizer 업데이트 ----
        std::set<std::string> trainable_params;
        for (const auto& op : E) {
            if (!op.param_id.empty())
                trainable_params.insert(op.param_id);
        }

        for (const auto& name : trainable_params) {
            auto t_it = tensors.find(name);
            auto g_it = gradients.find(name);
            auto s_it = shapes.find(name);
            if (t_it == tensors.end() || g_it == gradients.end() || s_it == shapes.end()) {
#ifdef GE_VERBOSE
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
#ifdef GE_DEBUG_SYNC
            CUDA_CHECK(cudaDeviceSynchronize());
#endif
        }

        // ---- grad 버퍼만 free ----
        std::unordered_set<const float*> freed;
        for (const auto& kv : gradients) {
            const float* p = kv.second;
            if (p && freed.insert(p).second) CUDA_CHECK(cudaFree(const_cast<float*>(p)));
        }
    }

    return loss;
}

// ===================== Pybind11 module =====================

PYBIND11_MODULE(graph_executor, m) {
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
        .value("SOFTMAX", OpType::SOFTMAX);

    py::enum_<OptimizerType>(m, "OptimizerType")
        .value("SGD", OptimizerType::SGD)
        .value("MOMENTUM", OptimizerType::MOMENTUM)
        .value("ADAM", OptimizerType::ADAM)
        .export_values();

    // Structs
    py::class_<OpExtraParams>(m, "OpExtraParams")
        .def(py::init([](){
            OpExtraParams e{};
            e.alpha = 0.01f;
            e.gelu_tanh = 1;
            e.temperature = 1.0f;
            e.axis = 1;
            return e;
        }))
        .def_readwrite("kernel_h", &OpExtraParams::kernel_h)
        .def_readwrite("kernel_w", &OpExtraParams::kernel_w)
        .def_readwrite("stride_h", &OpExtraParams::stride_h)
        .def_readwrite("stride_w", &OpExtraParams::stride_w)
        .def_readwrite("padding_h", &OpExtraParams::padding_h)
        .def_readwrite("padding_w", &OpExtraParams::padding_w)
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
        .def_readwrite("axis", &OpExtraParams::axis);

    py::class_<OpStruct>(m, "OpStruct")
        .def(py::init<>())
        .def(py::init<OpType, std::string, std::string, std::string, OpExtraParams>(),
             py::arg("op_type"), py::arg("input_id"), py::arg("param_id"),
             py::arg("output_id"), py::arg("extra_params"))
        .def_readwrite("op_type", &OpStruct::op_type)
        .def_readwrite("input_id", &OpStruct::input_id)
        .def_readwrite("param_id", &OpStruct::param_id)
        .def_readwrite("output_id", &OpStruct::output_id)
        .def_readwrite("extra_params", &OpStruct::extra_params);

    py::class_<Shape>(m, "Shape")
        .def(py::init<int, int>())
        .def_readwrite("rows", &Shape::rows)
        .def_readwrite("cols", &Shape::cols);

    // Graph APIs
    m.def("run_graph_forward_entry", &run_graph_forward_entry,
        py::arg("E"), py::arg("tensors"), py::arg("shapes"),
        py::arg("out_host"), py::arg("final_output_id"), py::arg("batch_size"));

    m.def("run_graph_with_loss_entry", &run_graph_with_loss_entry,
        py::arg("E"), py::arg("tensors"), py::arg("shapes"),
        py::arg("final_output_id"), py::arg("label_tensor_id"),
        py::arg("loss_type"), py::arg("batch_size"));

    m.def("run_graph_backward_entry", &run_graph_backward_entry,
        py::arg("E"), py::arg("tensors"), py::arg("shapes"),
        py::arg("gradients"), py::arg("final_output_id"), py::arg("batch_size"));

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
                reinterpret_cast<float*>(param_ptr),
                reinterpret_cast<const float*>(grad_ptr),
                reinterpret_cast<float*>(velocity_ptr),
                reinterpret_cast<float*>(m_ptr),
                reinterpret_cast<float*>(v_ptr),
#if AMSGRAD_ENABLE
                reinterpret_cast<float*>(vhat_max_ptr),
#endif
                lr, beta1, beta2, eps,
#if WEIGHT_DECAY_ENABLE
                weight_decay,
#endif
                size, opt_type, timestep
            );
            CUDA_CHECK(cudaGetLastError());
#ifdef GE_DEBUG_SYNC
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
                reinterpret_cast<float*>(param_ptr),
                reinterpret_cast<const float*>(grad_ptr),
                reinterpret_cast<float*>(velocity_ptr),
                reinterpret_cast<float*>(m_ptr),
                reinterpret_cast<float*>(v_ptr),
#if AMSGRAD_ENABLE
                reinterpret_cast<float*>(vhat_max_ptr),
#endif
                lr, beta1, beta2, eps,
#if WEIGHT_DECAY_ENABLE
                weight_decay,
#endif
                size, opt_type, timestep
            );
            CUDA_CHECK(cudaGetLastError());
#ifdef GE_DEBUG_SYNC
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
    );
}
