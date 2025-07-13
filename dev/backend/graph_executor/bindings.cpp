// bindings.cpp
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <string>

#include "run_graph.cuh"
#include "run_graph_backward.cuh"

namespace py = pybind11;

// ✅ Forward entry function (with batch_size)
void run_graph_entry(
    const std::vector<OpStruct>& E,
    const std::unordered_map<std::string, uintptr_t>& tensor_ptrs,
    const std::unordered_map<std::string, Shape>& shapes,
    py::array_t<float> out_host,
    const std::string& final_output_id,
    int batch_size)
{
    std::unordered_map<std::string, float*> tensors;
    for (const auto& kv : tensor_ptrs) {
        tensors[kv.first] = reinterpret_cast<float*>(kv.second);
    }

    float* out_ptr = out_host.mutable_data();
    run_graph_cuda(E, tensors, const_cast<std::unordered_map<std::string, Shape>&>(shapes), out_ptr, final_output_id, batch_size);
}

// ✅ Backward entry function (with batch_size)
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

    for (const auto& kv : tensor_ptrs) {
        tensors[kv.first] = reinterpret_cast<float*>(kv.second);
    }
    for (const auto& kv : gradient_ptrs) {
        gradients[kv.first] = reinterpret_cast<float*>(kv.second);
    }

    run_graph_backward(E, tensors, const_cast<std::unordered_map<std::string, Shape>&>(shapes), gradients, final_output_id, batch_size);

    py::dict result;
    for (const auto& kv : gradients) {
        result[py::str(kv.first)] = reinterpret_cast<uintptr_t>(kv.second);
    }

    return result;
}

// ✅ Pybind11 module definition
PYBIND11_MODULE(graph_executor, m) {
    py::class_<OpStruct>(m, "OpStruct")
        .def(py::init<int, std::string, std::string, std::string>())
        .def_readwrite("op_type", &OpStruct::op_type)
        .def_readwrite("input_id", &OpStruct::input_id)
        .def_readwrite("param_id", &OpStruct::param_id)
        .def_readwrite("output_id", &OpStruct::output_id);

    py::class_<Shape>(m, "Shape")
        .def(py::init<int, int>())
        .def_readwrite("rows", &Shape::rows)
        .def_readwrite("cols", &Shape::cols);

    m.def("run_graph_cuda", &run_graph_entry,
          py::arg("E"),
          py::arg("tensors"),
          py::arg("shapes"),
          py::arg("out_host"),
          py::arg("final_output_id"),
          py::arg("batch_size"));  // ✅ batch_size 추가

    m.def("run_graph_backward", &run_graph_backward_entry,
          py::arg("E"),
          py::arg("tensors"),
          py::arg("shapes"),
          py::arg("gradients"),
          py::arg("final_output_id"),
          py::arg("batch_size"));  // ✅ batch_size 추가
}
