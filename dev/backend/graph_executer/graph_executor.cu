#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // ✅ std::vector, std::string 등 자동 변환 지원

#include <vector>
#include <iostream>

namespace py = pybind11;

// ⭐ 실제 CUDA 커널이 있다면 여기 선언 (예: __global__ void kernel(...))
// 이 예시에서는 CPU mock 연산으로 대체합니다.

// ✅ run_graph 함수
void run_graph(
    py::array_t<int, py::array::forcecast> E_py,
    std::vector<py::array_t<float, py::array::forcecast>> params,
    std::vector<py::array_t<float, py::array::forcecast>> buffers,
    int input_id,
    int output_id
) {
    auto E_buf = E_py.unchecked<2>();
    const int num_ops = E_buf.shape(0);

    std::cout << "🚀 run_graph 시작 (총 연산 수: " << num_ops << ")\n";

    for (int i = 0; i < num_ops; ++i) {
        int op_type = E_buf(i, 0);
        int input_idx = E_buf(i, 1);
        int param_idx = E_buf(i, 2);
        int output_idx = E_buf(i, 3);

        auto input = buffers[input_idx].mutable_unchecked<2>();
        auto output = buffers[output_idx].mutable_unchecked<2>();

        if (op_type == 0) {  // matmul
            auto weight = params[-param_idx - 1000].unchecked<2>();
            int M = input.shape(0);
            int N = weight.shape(1);
            int K = weight.shape(0);

            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < N; ++n) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += input(m, k) * weight(k, n);
                    }
                    output(m, n) = sum;
                }
            }

        } else if (op_type == 1) {  // add (bias)
            auto bias = params[-param_idx - 1000].unchecked<2>();
            int M = input.shape(0);
            int N = input.shape(1);
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < N; ++n) {
                    output(m, n) = input(m, n) + bias(0, n);
                }
            }

        } else if (op_type == 2) {  // relu
            int M = input.shape(0);
            int N = input.shape(1);
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < N; ++n) {
                    output(m, n) = std::max(0.0f, input(m, n));
                }
            }

        } else if (op_type == 3) {  // sigmoid
            int M = input.shape(0);
            int N = input.shape(1);
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < N; ++n) {
                    float val = input(m, n);
                    output(m, n) = 1.0f / (1.0f + std::exp(-val));
                }
            }
        } else {
            std::cerr << "❌ 알 수 없는 연산 타입: " << op_type << "\n";
        }
    }

    std::cout << "✅ run_graph 완료\n";
}

// ✅ PYBIND11 모듈 정의
PYBIND11_MODULE(graph_executor, m) {
    m.doc() = "Graph Executor CUDA module";
    m.def("run_graph", &run_graph, "Run graph execution",
        py::arg("E_py"),
        py::arg("params"),
        py::arg("buffers"),
        py::arg("input_id"),
        py::arg("output_id")
    );

}
