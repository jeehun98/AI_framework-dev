#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "run_graph.cuh"

namespace py = pybind11;

void run_graph_cuda_py(
    py::array_t<int> E,
    py::array_t<int> shapes,
    py::array_t<float> W,
    py::array_t<float> b,
    py::array_t<float> x,
    py::array_t<float> out
) {
    run_graph_cuda(
        (int*)E.request().ptr, E.size(),
        (int*)shapes.request().ptr, shapes.size(),
        (float*)W.request().ptr, (float*)b.request().ptr,
        shapes.at(3), shapes.at(4),
        (float*)x.request().ptr, (float*)out.request().ptr
    );
}

PYBIND11_MODULE(graph_executor, m) {
    m.def("run_graph_cuda", &run_graph_cuda_py, "Run compiled graph on CUDA");
}
