#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/complex.h> // 추가: 필요에 따라
#include "../node/node.h" // Node class header

namespace py = pybind11;

// GPU data structure
struct NodeData {
    double input_value;
    double weight_value;
    double output;
    double bias;
};

// CUDA kernel: Matrix Addition
__global__ void matrix_add_cuda(
    const double* A, const double* B, double* C, size_t rows, size_t cols, 
    NodeData* node_list, bool is_new_graph
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
            node_list[idx] = {valueA, weight, sum, weight};
        } else {
            node_list[idx].input_value = valueA;
            node_list[idx].weight_value = weight;
            node_list[idx].output = sum;
            node_list[idx].bias = weight;
        }

        C[idx] = sum;
    }
}

// CUDA kernel: Matrix Multiplication
__global__ void matrix_multiply_cuda(
    const double* A, const double* B, double* C, size_t rows, size_t cols, size_t k, 
    NodeData* node_list, bool is_new_graph
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        double sum = 0.0;

        for (int i = 0; i < k; i++) {
            int indexA = row * k + i;
            int indexB = i * cols + col;

            double a_value = A[indexA];
            double b_value = B[indexB];

            double weight = is_new_graph ? b_value : node_list[idx].weight_value;
            double product = a_value * weight;

            sum += product;
        }

        if (is_new_graph) {
            node_list[idx] = {0.0, 0.0, sum, 0.0};
        } else {
            node_list[idx].output = sum;
        }

        C[idx] = sum;
    }
}

// Matrix Addition Function
std::pair<py::array_t<double>, std::vector<Node>> matrix_add(
    py::array_t<double> A, py::array_t<double> B, std::vector<Node> node_list
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
    NodeData* d_node_list;
    cudaMalloc(&d_A, rows * cols * sizeof(double));
    cudaMalloc(&d_B, rows * cols * sizeof(double));
    cudaMalloc(&d_C, rows * cols * sizeof(double));
    cudaMalloc(&d_node_list, rows * cols * sizeof(NodeData));

    cudaMemcpy(d_A, ptrA, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, ptrB, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    if (!node_list.empty()) {
        std::vector<NodeData> temp_node_list(node_list.size());
        for (size_t i = 0; i < node_list.size(); ++i) {
            temp_node_list[i] = {node_list[i].input_value, node_list[i].weight_value, 
                                 node_list[i].output, node_list[i].bias};
        }
        cudaMemcpy(d_node_list, temp_node_list.data(), rows * cols * sizeof(NodeData), cudaMemcpyHostToDevice);
    }

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);
    matrix_add_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols, d_node_list, node_list.empty());

    cudaMemcpy(ptrResult, d_C, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);

    if (node_list.empty()) {
        std::vector<NodeData> temp_node_list(rows * cols);
        cudaMemcpy(temp_node_list.data(), d_node_list, rows * cols * sizeof(NodeData), cudaMemcpyDeviceToHost);

        node_list.clear();
        for (const auto& nd : temp_node_list) {
            node_list.emplace_back("add", nd.input_value, nd.weight_value, nd.output, nd.bias);
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_node_list);

    return std::make_pair(result, node_list);
}

// Matrix Multiplication Function
std::pair<py::array_t<double>, std::vector<Node>> matrix_multiply(
    py::array_t<double> A, py::array_t<double> B, std::vector<Node> node_list
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
    NodeData* d_node_list;
    cudaMalloc(&d_A, rows * k * sizeof(double));
    cudaMalloc(&d_B, k * cols * sizeof(double));
    cudaMalloc(&d_C, rows * cols * sizeof(double));
    cudaMalloc(&d_node_list, rows * cols * sizeof(NodeData));

    cudaMemcpy(d_A, ptrA, rows * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, ptrB, k * cols * sizeof(double), cudaMemcpyHostToDevice);
    if (!node_list.empty()) {
        std::vector<NodeData> temp_node_list(node_list.size());
        for (size_t i = 0; i < node_list.size(); ++i) {
            temp_node_list[i] = {node_list[i].input_value, node_list[i].weight_value, 
                                 node_list[i].output, node_list[i].bias};
        }
        cudaMemcpy(d_node_list, temp_node_list.data(), rows * cols * sizeof(NodeData), cudaMemcpyHostToDevice);
    }

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);
    matrix_multiply_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols, k, d_node_list, node_list.empty());

    cudaMemcpy(ptrResult, d_C, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);

    if (node_list.empty()) {
        std::vector<NodeData> temp_node_list(rows * cols);
        cudaMemcpy(temp_node_list.data(), d_node_list, rows * cols * sizeof(NodeData), cudaMemcpyHostToDevice);

        node_list.clear();
        for (const auto& nd : temp_node_list) {
            node_list.emplace_back("multiply", nd.input_value, nd.weight_value, nd.output, nd.bias);
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_node_list);

    return std::make_pair(result, node_list);
}
// CUDA 바인딩 정의
PYBIND11_MODULE(operations_matrix_cuda, m) {
    // Node 클래스 바인딩
    py::class_<Node, std::shared_ptr<Node>>(m, "Node")
        .def(py::init<const std::string&, double, double, double, double>())
        .def_readwrite("operation", &Node::operation)
        .def_readwrite("input_value", &Node::input_value)
        .def_readwrite("weight_value", &Node::weight_value)
        .def_readwrite("output", &Node::output)
        .def_readwrite("bias", &Node::bias);

    // std::vector<std::shared_ptr<Node>> 바인딩
    py::bind_vector<std::vector<std::shared_ptr<Node>>>(m, "NodeList");

    // matrix_add 및 matrix_multiply 함수 바인딩
    m.def("matrix_add", &matrix_add,
          py::arg("A"), py::arg("B"), py::arg("node_list") = std::vector<std::shared_ptr<Node>>(),
          py::return_value_policy::move);
    m.def("matrix_multiply", &matrix_multiply,
          py::arg("A"), py::arg("B"), py::arg("node_list") = std::vector<std::shared_ptr<Node>>(),
          py::return_value_policy::move);
}