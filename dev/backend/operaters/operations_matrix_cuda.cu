#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <memory>
#include "../node/node.h"  // Node 클래스가 정의된 헤더 파일 포함

namespace py = pybind11;

// CUDA 커널: 행렬 덧셈
__global__ void matrix_add_cuda(
    const double* A, const double* B, double* C, size_t rows, size_t cols, 
    Node* node_list, bool is_new_graph
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        double valueA = A[idx];
        double valueB = B[idx];

        double weight = is_new_graph ? valueB : node_list[idx].weight_value;
        double sum = valueA + weight;

        if (is_new_graph) {
            node_list[idx] = Node("add", valueA, weight, sum, weight);
        } else {
            node_list[idx].update(valueA, weight, sum, weight);
        }

        C[idx] = sum;
    }
}

// CUDA 커널: 행렬 곱셈
__global__ void matrix_multiply_cuda(
    const double* A, const double* B, double* C, size_t rows, size_t cols, size_t k, 
    Node* node_list, bool is_new_graph
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        double sum = 0.0;

        Node sum_node;
        if (is_new_graph) {
            sum_node = Node("add", 0.0, 0.0, 0.0, 0.0);
        } else {
            sum_node = node_list[idx];
            sum_node.output = 0.0;
        }

        for (int i = 0; i < k; i++) {
            int indexA = row * k + i;
            int indexB = i * cols + col;

            double a_value = A[indexA];
            double b_value = B[indexB];

            double weight = is_new_graph ? b_value : sum_node.children[i]->weight_value;
            double product = a_value * weight;

            if (is_new_graph) {
                Node mul_node = Node("multiply", a_value, weight, product, weight);
                sum_node.add_child(mul_node);
                mul_node.add_parent(sum_node);
            } else {
                Node* mul_node = sum_node.get_children()[i];
                mul_node->update(a_value, weight, product, weight);
            }

            sum += product;
        }

        sum_node.output = sum;
        node_list[idx] = sum_node;

        C[idx] = sum;
    }
}

// 행렬 덧셈 함수
std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> matrix_add(
    py::array_t<double> A, py::array_t<double> B, std::vector<std::shared_ptr<Node>> node_list
) {
    py::buffer_info bufA = A.request();
    py::buffer_info bufB = B.request();

    size_t rows = bufA.shape[0];
    size_t cols = bufA.shape[1];

    double* ptrA = static_cast<double*>(bufA.ptr);
    double* ptrB = static_cast<double*>(bufB.ptr);

    py::array_t<double> result({rows, cols});
    py::buffer_info bufResult = result.request();
    double* ptrResult = static_cast<double*>(bufResult.ptr);

    double *d_A, *d_B, *d_C;
    Node *d_node_list;
    cudaMalloc(&d_A, rows * cols * sizeof(double));
    cudaMalloc(&d_B, rows * cols * sizeof(double));
    cudaMalloc(&d_C, rows * cols * sizeof(double));
    cudaMalloc(&d_node_list, rows * cols * sizeof(Node));

    cudaMemcpy(d_A, ptrA, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, ptrB, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    if (!node_list.empty()) {
        cudaMemcpy(d_node_list, node_list.data(), rows * cols * sizeof(Node), cudaMemcpyHostToDevice);
    }

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);
    matrix_add_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols, d_node_list, node_list.empty());

    cudaMemcpy(ptrResult, d_C, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
    if (node_list.empty()) {
        node_list.resize(rows * cols);
        cudaMemcpy(node_list.data(), d_node_list, rows * cols * sizeof(Node), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_node_list);

    return std::make_pair(result, node_list);
}

// 행렬 곱셈 함수
std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> matrix_multiply(
    py::array_t<double> A, py::array_t<double> B, std::vector<std::shared_ptr<Node>> node_list
) {
    py::buffer_info bufA = A.request();
    py::buffer_info bufB = B.request();

    size_t rows = bufA.shape[0];
    size_t k = bufA.shape[1];
    size_t cols = bufB.shape[1];

    double* ptrA = static_cast<double*>(bufA.ptr);
    double* ptrB = static_cast<double*>(bufB.ptr);

    py::array_t<double> result({rows, cols});
    py::buffer_info bufResult = result.request();
    double* ptrResult = static_cast<double*>(bufResult.ptr);

    double *d_A, *d_B, *d_C;
    Node *d_node_list;
    cudaMalloc(&d_A, rows * k * sizeof(double));
    cudaMalloc(&d_B, k * cols * sizeof(double));
    cudaMalloc(&d_C, rows * cols * sizeof(double));
    cudaMalloc(&d_node_list, rows * cols * sizeof(Node));

    cudaMemcpy(d_A, ptrA, rows * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, ptrB, k * cols * sizeof(double), cudaMemcpyHostToDevice);
    if (!node_list.empty()) {
        cudaMemcpy(d_node_list, node_list.data(), rows * cols * sizeof(Node), cudaMemcpyHostToDevice);
    }

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);
    matrix_multiply_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols, k, d_node_list, node_list.empty());

    cudaMemcpy(ptrResult, d_C, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
    if (node_list.empty()) {
        node_list.resize(rows * cols);
        cudaMemcpy(node_list.data(), d_node_list, rows * cols * sizeof(Node), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_node_list);

    return std::make_pair(result, node_list);
}

// Pybind11 모듈 정의
PYBIND11_MODULE(operations_matrix_cuda, m) {
    m.doc() = "CUDA accelerated matrix operations with computation graph support";

    m.def("matrix_add", &matrix_add, 
          py::arg("A"), py::arg("B"), py::arg("node_list") = std::vector<std::shared_ptr<Node>>(), 
          "CUDA matrix addition with optional node_list");

    m.def("matrix_multiply", &matrix_multiply, 
          py::arg("A"), py::arg("B"), py::arg("node_list") = std::vector<std::shared_ptr<Node>>(), 
          "CUDA matrix multiplication with optional node_list");
}
