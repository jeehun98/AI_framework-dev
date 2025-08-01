#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <string>
#include <iostream>

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
        .def_readwrite("use_bias", &OpExtraParams::use_bias);

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
        optimizer_update_cuda(
            reinterpret_cast<float*>(param_ptr),
            reinterpret_cast<float*>(grad_ptr),
            reinterpret_cast<float*>(velocity_ptr),
            reinterpret_cast<float*>(m_ptr),
            reinterpret_cast<float*>(v_ptr),
            lr, beta1, beta2, eps, size, opt_type, timestep);
    }, py::arg("param_ptr"), py::arg("grad_ptr"),
       py::arg("velocity_ptr") = 0, py::arg("m_ptr") = 0, py::arg("v_ptr") = 0,
       py::arg("lr") = 0.01, py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
       py::arg("eps") = 1e-8, py::arg("size"), py::arg("opt_type"),
       py::arg("timestep") = 1);
}
