#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "run_graph.cuh"

namespace py = pybind11;

void run_graph_wrapper(
    py::array_t<int> E,
    int E_len,
    py::array_t<int> shapes,
    int shapes_len,
    py::array_t<float> W,
    py::array_t<float> b,
    int W_rows,
    int W_cols
) {
    auto E_ptr = static_cast<int*>(E.request().ptr);
    auto shapes_ptr = static_cast<int*>(shapes.request().ptr);
    auto W_ptr = static_cast<float*>(W.request().ptr);
    auto b_ptr = static_cast<float*>(b.request().ptr);

    run_graph_cuda(E_ptr, E_len, shapes_ptr, shapes_len, W_ptr, b_ptr, W_rows, W_cols);
}

PYBIND11_MODULE(graph_executor, m) {
    m.def("run_graph_cuda", &run_graph_wrapper,
        py::arg("E"),
        py::arg("E_len"),
        py::arg("shapes"),
        py::arg("shapes_len"),
        py::arg("W"),
        py::arg("b"),
        py::arg("W_rows"),
        py::arg("W_cols")
    );
}
