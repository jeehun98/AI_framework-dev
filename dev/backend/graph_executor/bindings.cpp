// bindings.cpp
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <string>
#include <iostream>
#include <stdio.h>  // CUDA printf 용

#include "run_graph.cuh"
#include "run_graph_backward.cuh"
#include "run_graph_with_loss.cuh"
#include "op_structs.cuh"
#include "optimizer_types.cuh"   // enum class OptimizerType

namespace py = pybind11;

// ===== 매크로 기본값(없으면 0) =====
#ifndef WEIGHT_DECAY_ENABLE
#define WEIGHT_DECAY_ENABLE 0
#endif
#ifndef AMSGRAD_ENABLE
#define AMSGRAD_ENABLE 0
#endif
#ifndef GLOBAL_NORM_CLIP_ENABLE
#define GLOBAL_NORM_CLIP_ENABLE 0
#endif

#include "optimizer_config.cuh"
#include "optimizer_kernels.cuh"

// ===== 안전 캐스터 (int -> OptimizerType) =====
static inline OptimizerType to_opt(int v) {
    switch (v) {
        case 0: return OptimizerType::SGD;
        case 1: return OptimizerType::MOMENTUM;
        case 2: return OptimizerType::ADAM;
        default: return OptimizerType::ADAM;
    }
}

// =========================== 엔트리 함수들 ===========================

// Forward-only
void run_graph_forward_entry(
    const std::vector<OpStruct>& E,
    const std::unordered_map<std::string, uintptr_t>& tensor_ptrs,
    const std::unordered_map<std::string, Shape>& shapes,
    py::array_t<float> out_host,
    const std::string& final_output_id,
    int batch_size)
{
    std::unordered_map<std::string, float*> tensors;
    tensors.reserve(tensor_ptrs.size());
    for (const auto& kv : tensor_ptrs) {
        tensors[kv.first] = reinterpret_cast<float*>(kv.second);
    }

    float* out_ptr = out_host.mutable_data();
    run_graph_cuda(E, tensors,
                   const_cast<std::unordered_map<std::string, Shape>&>(shapes),
                   out_ptr, final_output_id, batch_size);
}

// Forward + Loss
float run_graph_with_loss_entry(
    const std::vector<OpStruct>& E,
    const std::unordered_map<std::string, uintptr_t>& tensor_ptrs,
    const std::unordered_map<std::string, Shape>& shapes,
    const std::string& final_output_id,
    const std::string& label_tensor_id,
    const std::string& loss_type,
    int batch_size)
{
    std::unordered_map<std::string, float*> tensors;
    tensors.reserve(tensor_ptrs.size());
    for (const auto& kv : tensor_ptrs) {
        tensors[kv.first] = reinterpret_cast<float*>(kv.second);
    }

    return run_graph_with_loss_cuda(E, tensors,
            const_cast<std::unordered_map<std::string, Shape>&>(shapes),
            final_output_id, label_tensor_id, loss_type, batch_size);
}

// Backward
py::dict run_graph_backward_entry(
    const std::vector<OpStruct>& E,
    const std::unordered_map<std::string, uintptr_t>& tensor_ptrs,
    const std::unordered_map<std::string, Shape>& shapes_in,
    const std::unordered_map<std::string, uintptr_t>& /*gradient_ptrs - unused*/,
    const std::string& final_output_id,
    int batch_size)
{
    // 1) 포인터 맵 구성
    std::unordered_map<std::string, float*> tensors;
    for (const auto& kv : tensor_ptrs)
        tensors[kv.first] = reinterpret_cast<float*>(kv.second);

    // shapes는 내부 함수 시그니처 때문에 const_cast
    auto& shapes = const_cast<std::unordered_map<std::string, Shape>&>(shapes_in);

    // 2) ✅ 먼저 forward를 한 번 돌려, LOSS 직전 출력(pred_id)을 디바이스에 생성/보존
    std::string pred_id = final_output_id;
    if (!E.empty() && E.back().op_type == OpType::LOSS) {
        pred_id = E.back().input_id;   // LOSS 입력이 y_pred
    }
    // out_host=nullptr → 디바이스 출력을 tensors에만 보존
    run_graph_cuda(E, tensors, shapes, /*out_host=*/nullptr, pred_id, batch_size);

    // 3) backward 실행 (gradients는 내부에서 새로 할당)
    std::unordered_map<std::string, float*> gradients;
    run_graph_backward(E, tensors, shapes, gradients, final_output_id, batch_size);

    // 4) 파이썬으로 gradient 포인터들 반환
    py::dict result;
    for (const auto& kv : gradients)
        result[py::str(kv.first)] = reinterpret_cast<uintptr_t>(kv.second);

    return result;
}

// Train step (Fwd+Loss -> Bwd -> Optimizer)
float train_step_entry(
    const std::vector<OpStruct>& E,
    const std::unordered_map<std::string, uintptr_t>& tensor_ptrs,
    const std::unordered_map<std::string, Shape>& shapes_in,
    const std::string& final_output_id,
    const std::string& label_tensor_id,
    const std::string& loss_type,
    int batch_size,
    // Optimizer
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
    // 1) 텐서/셰이프 맵 준비
    std::unordered_map<std::string, float*> tensors;
    tensors.reserve(tensor_ptrs.size());
    for (const auto& kv : tensor_ptrs) {
        tensors[kv.first] = reinterpret_cast<float*>(kv.second);
    }
    auto& shapes = const_cast<std::unordered_map<std::string, Shape>&>(shapes_in);

    // 2) Forward + Loss
    float loss = run_graph_with_loss_cuda(
        E, tensors, shapes, final_output_id, label_tensor_id, loss_type, batch_size);

    // 3) Backward
    std::unordered_map<std::string, float*> gradients;
    run_graph_backward(E, tensors, shapes, gradients, final_output_id, batch_size);

    // 4) 학습 대상(파라미터) 집합 구성
    std::set<std::string> trainable_params;
    for (const auto& op : E) {
        const bool uses_param =
            (op.op_type == OpType::MATMUL) ||
            (op.op_type == OpType::ADD)    ||
            (op.op_type == OpType::CONV2D);
        if (uses_param && !op.param_id.empty()) {
            trainable_params.insert(op.param_id);
        }
    }

    // 5) Optimizer update
    for (const auto& name : trainable_params) {
        auto t_it = tensors.find(name);
        auto g_it = gradients.find(name);
        auto s_it = shapes.find(name);
        if (t_it == tensors.end() || g_it == gradients.end() || s_it == shapes.end()) {
            continue; // 없는 경우 skip
        }

        float*       param_ptr = t_it->second;
        const float* grad_ptr  = g_it->second;
        const Shape& shp       = s_it->second;
        int size = shp.rows * shp.cols;

        uintptr_t vel_u = 0, m_u = 0, v_u = 0;
#if AMSGRAD_ENABLE
        uintptr_t vhat_u = 0;
#endif
    // C++14 호환
    auto it_vel = velocity_ptrs.find(name);
    if (it_vel != velocity_ptrs.end()) vel_u = it_vel->second;

    auto it_m = m_ptrs.find(name);
    if (it_m != m_ptrs.end()) m_u = it_m->second;

    auto it_v = v_ptrs.find(name);
    if (it_v != v_ptrs.end()) v_u = it_v->second;

#if AMSGRAD_ENABLE
    auto it_vhat = vhat_max_ptrs.find(name);
    if (it_vhat != vhat_max_ptrs.end()) vhat_u = it_vhat->second;
#endif


        optimizer_update_cuda(
            param_ptr,
            grad_ptr,
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
    }

    // 6) 임시 grad 메모리 정리(중복 free 방지)
    std::unordered_set<float*> freed;
    for (const auto& kv : gradients) {
        float* p = kv.second;
        if (!p) continue;
        if (freed.insert(p).second) cudaFree(p);
    }
    return loss;
}

// =========================== Pybind11 모듈 ===========================

PYBIND11_MODULE(graph_executor, m) {
    // ----- Enums -----
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

    // ----- Structs -----
    py::class_<OpExtraParams>(m, "OpExtraParams")
        .def(py::init([](){
            OpExtraParams e;
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
        // 가능하면 OpType을 직접 받도록:
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

    // ----- Graph APIs -----
    m.def("run_graph_forward_entry", &run_graph_forward_entry,
        py::arg("E"), py::arg("tensors"), py::arg("shapes"),
        py::arg("out_host"), py::arg("final_output_id"), py::arg("batch_size"));

    m.def("run_graph_with_loss_entry",
        [](const std::vector<OpStruct>& E,
           const std::unordered_map<std::string, uintptr_t>& tensor_ints,
           std::unordered_map<std::string, Shape>& shapes,
           const std::string& final_output_id,
           const std::string& label_tensor_id,
           const std::string& loss_type,
           int batch_size) -> float {
            std::unordered_map<std::string, float*> tensors;
            tensors.reserve(tensor_ints.size());
            for (auto& kv : tensor_ints) {
                tensors[kv.first] = reinterpret_cast<float*>(kv.second);
            }
            return run_graph_with_loss_cuda(
                E, tensors, shapes, final_output_id, label_tensor_id, loss_type, batch_size
            );
        },
        py::arg("E"), py::arg("tensors"), py::arg("shapes"),
        py::arg("final_output_id"), py::arg("label_tensor_id"),
        py::arg("loss_type"), py::arg("batch_size"));

    m.def("run_graph_backward_entry", &run_graph_backward_entry,
        py::arg("E"), py::arg("tensors"), py::arg("shapes"),
        py::arg("gradients"), py::arg("final_output_id"), py::arg("batch_size"));

    // ----- Optimizer: enum 버전 -----
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
        ) {
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

    // ----- Optimizer: int 버전(레거시 호환용) -----
    m.def("optimizer_update",
    [](uintptr_t param_ptr, uintptr_t grad_ptr,
    uintptr_t velocity_ptr, uintptr_t m_ptr, uintptr_t v_ptr,
    #if WEIGHT_DECAY_ENABLE
    float weight_decay,
    #endif
    float lr, float beta1, float beta2, float eps,
    int size, OptimizerType opt_type, int timestep   // ← 여기 opt_type 포함!
    #if AMSGRAD_ENABLE
    ,uintptr_t vhat_max_ptr
    #endif
    ){
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
            size,                 // 1) size
            opt_type,             // 2) OptimizerType
            timestep              // 3) t
        );
    },
    py::arg("param_ptr"), py::arg("grad_ptr"),
    py::arg("velocity_ptr") = 0, py::arg("m_ptr") = 0, py::arg("v_ptr") = 0,
    #if WEIGHT_DECAY_ENABLE
    py::arg("weight_decay") = 0.0f,
    #endif
    py::arg("lr") = 0.01f, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f,
    py::arg("size"),
    py::arg("opt_type"),                 // ← 이 이름과 람다 인자명이 일치해야 함
    py::arg("timestep") = 1
    #if AMSGRAD_ENABLE
    ,py::arg("vhat_max_ptr") = 0
    #endif
    );


    // ----- One-shot train step -----
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
