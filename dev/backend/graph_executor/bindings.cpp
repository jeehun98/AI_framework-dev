#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "run_graph.cuh"

namespace py = pybind11;

py::array_t<float> run_graph_cuda_wrapper(
    py::array_t<int> E,
    int E_len,
    py::array_t<int> shapes,
    int shapes_len,
    py::array_t<float> W,
    py::array_t<float> b,
    int W_rows,
    int W_cols,
    int activation_type)  // ✅ 추가
{
    auto result = py::array_t<float>({shapes.at(1), W_cols});
    float* out_ptr;
    cudaMallocHost(&out_ptr, sizeof(float) * shapes.at(1) * W_cols);

    run_graph_cuda(
        E.mutable_data(), E_len,
        shapes.mutable_data(), shapes_len,
        W.mutable_data(), b.mutable_data(),
        W_rows, W_cols, activation_type,  // ✅ 추가
        out_ptr);

    std::memcpy(result.mutable_data(), out_ptr, sizeof(float) * shapes.at(1) * W_cols);
    cudaFreeHost(out_ptr);
    return result;
}

PYBIND11_MODULE(graph_executor, m) {
    m.def("run_graph_cuda", &run_graph_cuda_wrapper,
          py::arg("E"), py::arg("E_len"),
          py::arg("shapes"), py::arg("shapes_len"),
          py::arg("W"), py::arg("b"),
          py::arg("W_rows"), py::arg("W_cols"),
          py::arg("activation_type"));  // ✅ 추가
}
