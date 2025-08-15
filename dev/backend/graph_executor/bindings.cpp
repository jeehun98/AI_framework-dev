#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <string>
#include <iostream>
#include <stdio.h>  // ✅ CUDA printf 사용을 위해 필요


#include "run_graph.cuh"
#include "run_graph_backward.cuh"
#include "run_graph_with_loss.cuh"
#include "op_structs.cuh"
#include "optimizer_kernels.cuh"
#include "optimizer_types.cuh"

namespace py = pybind11;

// ✅ Forward-only 실행 함수
void run_graph_forward_entry(
    const std::vector<OpStruct>& E,
    const std::unordered_map<std::string, uintptr_t>& tensor_ptrs,
    const std::unordered_map<std::string, Shape>& shapes,
    py::array_t<float> out_host,
    const std::string& final_output_id,
    int batch_size)
{
    std::unordered_map<std::string, float*> tensors;
    for (const auto& kv : tensor_ptrs)
        tensors[kv.first] = reinterpret_cast<float*>(kv.second);

    float* out_ptr = out_host.mutable_data();
    run_graph_cuda(E, tensors, const_cast<std::unordered_map<std::string, Shape>&>(shapes), out_ptr, final_output_id, batch_size);
}

// ✅ Forward + Loss 계산
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
    for (const auto& kv : tensor_ptrs)
        tensors[kv.first] = reinterpret_cast<float*>(kv.second);

    return run_graph_with_loss_cuda(E, tensors,
        const_cast<std::unordered_map<std::string, Shape>&>(shapes),
        final_output_id, label_tensor_id, loss_type, batch_size);
}

// ✅ Backward 계산
py::dict run_graph_backward_entry(
    const std::vector<OpStruct>& E,
    const std::unordered_map<std::string, uintptr_t>& tensor_ptrs,
    const std::unordered_map<std::string, Shape>& shapes,
    const std::unordered_map<std::string, uintptr_t>& gradient_ptrs,
    const std::string& final_output_id,
    int batch_size)
{
    std::unordered_map<std::string, float*> tensors;
    std::unordered_map<std::string, float*> gradients;

    for (const auto& kv : tensor_ptrs)
        tensors[kv.first] = reinterpret_cast<float*>(kv.second);
    for (const auto& kv : gradient_ptrs)
        gradients[kv.first] = reinterpret_cast<float*>(kv.second);

    run_graph_backward(E, tensors, const_cast<std::unordered_map<std::string, Shape>&>(shapes),
                       gradients, final_output_id, batch_size);

    py::dict result;
    for (const auto& kv : gradients)
        result[py::str(kv.first)] = reinterpret_cast<uintptr_t>(kv.second);

    return result;
}

// ✅ 한 번에 학습(Forward+Loss → Backward → Optimizer Update)
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
    // (선택) 옵티마이저 상태 버퍼들: 파라미터 이름 → 디바이스 포인터
    const std::unordered_map<std::string, uintptr_t>& velocity_ptrs = {},
    const std::unordered_map<std::string, uintptr_t>& m_ptrs = {},
    const std::unordered_map<std::string, uintptr_t>& v_ptrs = {})
{
    // 1) 텐서/셰이프 맵 준비
    std::unordered_map<std::string, float*> tensors;
    for (const auto& kv : tensor_ptrs)
        tensors[kv.first] = reinterpret_cast<float*>(kv.second);

    // const_cast: 내부 커널 시그니처 유지
    auto& shapes = const_cast<std::unordered_map<std::string, Shape>&>(shapes_in);

    // 2) Forward + Loss
    float loss = run_graph_with_loss_cuda(
        E, tensors, shapes, final_output_id, label_tensor_id, loss_type, batch_size);

    // 3) Backward (gradients에 파라미터/중간 그래디언트들이 채워짐)
    std::unordered_map<std::string, float*> gradients;
    run_graph_backward(E, tensors, shapes, gradients, final_output_id, batch_size);

    // 4) 학습 대상(파라미터) 집합 구성: E에서 param_id를 가진 연산만 추림
    std::set<std::string> trainable_params;
    for (const auto& op : E) {
        // 연산 종류별 파라미터 존재 여부
        const bool uses_param =
            (op.op_type == OpType::MATMUL) ||
            (op.op_type == OpType::ADD)    ||
            (op.op_type == OpType::CONV2D);
        if (uses_param && !op.param_id.empty())
            trainable_params.insert(op.param_id);
    }

    // 5) Optimizer update
    for (const auto& name : trainable_params) {
        // 파라미터/그래디언트 포인터와 사이즈 확인
        auto t_it = tensors.find(name);
        auto g_it = gradients.find(name);
        auto s_it = shapes.find(name);
        if (t_it == tensors.end() || g_it == gradients.end() || s_it == shapes.end()) {
            // 그래디언트가 없을 수 있음(해당 step에서 사용X 등) → 건너뜀
            continue;
        }

        float* param_ptr = t_it->second;
        float* grad_ptr  = g_it->second;
        const Shape& shp = s_it->second;
        int size = shp.rows * shp.cols;

        // 옵티마이저 상태 버퍼(없으면 0)
        uintptr_t v_ptr_u = 0, m_ptr_u = 0, vel_ptr_u = 0;

        // 수정 (C++14 호환)
        auto it_v = v_ptrs.find(name);
        if (it_v != v_ptrs.end()) v_ptr_u = it_v->second;

        auto it_m = m_ptrs.find(name);
        if (it_m != m_ptrs.end()) m_ptr_u = it_m->second;

        auto it_vel = velocity_ptrs.find(name);
        if (it_vel != velocity_ptrs.end()) vel_ptr_u = it_vel->second;

        optimizer_update_cuda(
            param_ptr,
            reinterpret_cast<const float*>(grad_ptr),
            reinterpret_cast<float*>(vel_ptr_u),
            reinterpret_cast<float*>(m_ptr_u),
            reinterpret_cast<float*>(v_ptr_u),
            lr, beta1, beta2, eps, size, opt_type, timestep
        );
    }

    // 6) 임시 grad 메모리 정리(중복 free 방지)
    std::unordered_set<float*> freed;
    for (const auto& kv : gradients) {
        float* p = kv.second;
        if (!p) continue;
        if (freed.insert(p).second) {
            cudaFree(p);
        }
    }

    return loss;
}

// ✅ Pybind11 모듈 정의
PYBIND11_MODULE(graph_executor, m) {
    // 🔷 Enum 정의
    py::enum_<OpType>(m, "OpType")
        .value("MATMUL", OpType::MATMUL)
        .value("ADD", OpType::ADD)
        .value("RELU", OpType::RELU)
        .value("SIGMOID", OpType::SIGMOID)
        .value("TANH", OpType::TANH)
        .value("FLATTEN", OpType::FLATTEN)
        .value("CONV2D", OpType::CONV2D)
        .value("LOSS", OpType::LOSS);

    py::enum_<OptimizerType>(m, "OptimizerType")
        .value("SGD", OptimizerType::SGD)
        .value("MOMENTUM", OptimizerType::MOMENTUM)
        .value("ADAM", OptimizerType::ADAM);

    // 🔷 구조체 바인딩
    py::class_<OpStruct>(m, "OpStruct")
        .def(py::init<>())
        .def(py::init<int, std::string, std::string, std::string, OpExtraParams>(),
             py::arg("op_type"), py::arg("input_id"), py::arg("param_id"),
             py::arg("output_id"), py::arg("extra_params"))
        .def_readwrite("op_type", &OpStruct::op_type)
        .def_readwrite("input_id", &OpStruct::input_id)
        .def_readwrite("param_id", &OpStruct::param_id)
        .def_readwrite("output_id", &OpStruct::output_id)
        .def_readwrite("extra_params", &OpStruct::extra_params);

    py::class_<OpExtraParams>(m, "OpExtraParams")
        .def(py::init<>())
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
        .def_readwrite("loss_type", &OpExtraParams::loss_type);

    py::class_<Shape>(m, "Shape")
        .def(py::init<int, int>())
        .def_readwrite("rows", &Shape::rows)
        .def_readwrite("cols", &Shape::cols);

    

    // 🔷 그래프 관련 함수
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

    // 🔷 Optimizer 함수 바인딩
    m.def("optimizer_update", [](uintptr_t param_ptr, uintptr_t grad_ptr,
                                 uintptr_t velocity_ptr, uintptr_t m_ptr, uintptr_t v_ptr,
                                 float lr, float beta1, float beta2, float eps,
                                 int size, OptimizerType opt_type, int timestep) {

        // ✅ 여기에 디버깅 로그 추가
        // printf("[PYBIND] Optimizer Update → lr=%.6f, beta1=%.4f, beta2=%.4f, eps=%.1e, size=%d, timestep=%d, opt=%d\n", lr, beta1, beta2, eps, size, timestep, static_cast<int>(opt_type));


        optimizer_update_cuda(
            reinterpret_cast<float*>(param_ptr),
            reinterpret_cast<const float*>(grad_ptr),
            reinterpret_cast<float*>(velocity_ptr),
            reinterpret_cast<float*>(m_ptr),
            reinterpret_cast<float*>(v_ptr),
            lr, beta1, beta2, eps, size, opt_type, timestep);
    }, py::arg("param_ptr"), py::arg("grad_ptr"),
       py::arg("velocity_ptr") = 0, py::arg("m_ptr") = 0, py::arg("v_ptr") = 0,
       py::arg("lr") = 0.01, py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
       py::arg("eps") = 1e-8, py::arg("size"), py::arg("opt_type"),
       py::arg("timestep") = 1);

    // ✅ 한 번에 학습 엔트리 바인딩
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
        py::arg("velocity_ptrs") = std::unordered_map<std::string, uintptr_t>{},
        py::arg("m_ptrs") = std::unordered_map<std::string, uintptr_t>{},
        py::arg("v_ptrs") = std::unordered_map<std::string, uintptr_t>{}
    );
}
