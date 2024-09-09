#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include "../node/node.h"  // Node 클래스가 정의된 헤더 파일 포함

namespace py = pybind11;

// 배치 단위 노드
std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> matrix_add(py::array_t<double> A, py::array_t<double> B) {
    // Numpy 배열의 버퍼 정보 가져오기
    py::buffer_info bufA = A.request(), bufB = B.request();

    // 입력 배열이 3D인지 확인 (배치 크기 x 행 크기 x 열 크기)
    if (bufA.ndim != 3 || bufB.ndim != 3) {
        throw std::runtime_error("Input should be 3-D NumPy arrays");
    }

    // 배치 크기, 행과 열의 크기 확인
    if (bufA.shape[0] != bufB.shape[0] || bufA.shape[1] != bufB.shape[1] || bufA.shape[2] != bufB.shape[2]) {
        throw std::runtime_error("Input matrices must have the same shape");
    }

    // 배치 크기, 행과 열의 크기
    size_t batch_size = bufA.shape[0];
    size_t rows = bufA.shape[1];
    size_t cols = bufA.shape[2];

    // 결과 배열 생성
    py::array_t<double> result = py::array_t<double>({batch_size, rows, cols});
    py::buffer_info bufResult = result.request();

    double* ptrA = static_cast<double*>(bufA.ptr);
    double* ptrB = static_cast<double*>(bufB.ptr);
    double* ptrResult = static_cast<double*>(bufResult.ptr);

    // 노드 리스트 생성
    std::vector<std::shared_ptr<Node>> node_list;

    // 배치 단위로 행렬 덧셈 및 노드 생성
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                size_t index = b * rows * cols + i * cols + j;
                double valueA = ptrA[index];
                double valueB = ptrB[index];
                double sum = valueA + valueB;

                // 덧셈 노드 생성
                std::shared_ptr<Node> add_node = std::make_shared<Node>("add", valueA, valueB, sum);
                node_list.push_back(add_node);

                // 결과 저장
                ptrResult[index] = sum;
            }
        }
    }

    // 결과 배열과 노드 리스트 반환
    return std::make_pair(result, node_list);
}

// 입력 데이터는 3차원 (배치, 행, 열), 가중치는 보통 배치 단위가 아닌 (행, 열) 의 사이즈
std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> matrix_multiply(py::array_t<double> A, py::array_t<double> B, std::vector<std::shared_ptr<Node>> node_list = {}) {
    // Numpy 배열의 버퍼 정보 가져오기
    py::buffer_info bufA = A.request(), bufB = B.request();

    // A는 3D, B는 2D 배열인지 확인
    if (bufA.ndim != 3 || bufB.ndim != 2) {
        throw std::runtime_error("Input A should be a 3-D NumPy array and B should be a 2-D NumPy array");
    }

    // A와 B의 차원 체크
    if (bufA.shape[2] != bufB.shape[0]) {
        throw std::runtime_error("The number of features in A must match the number of rows in B");
    }

    // 배치 크기, 행과 열의 크기
    size_t batch_size = bufA.shape[0];
    size_t input_dim = bufA.shape[1];
    size_t feature_dim = bufA.shape[2];
    size_t output_dim = bufB.shape[1];

    // 결과 배열 생성
    py::array_t<double> result = py::array_t<double>({batch_size, input_dim, output_dim});
    py::buffer_info bufResult = result.request();

    double* ptrA = static_cast<double*>(bufA.ptr);
    double* ptrB = static_cast<double*>(bufB.ptr);
    double* ptrResult = static_cast<double*>(bufResult.ptr);

    // 행렬 곱셈
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < input_dim; ++i) {
            for (size_t j = 0; j < output_dim; ++j) {
                size_t result_index = b * input_dim * output_dim + i * output_dim + j;
                ptrResult[result_index] = 0;

                // 덧셈 노드 생성
                std::shared_ptr<Node> sum_node = std::make_shared<Node>("sum", 0, 0, 0.0);
                node_list.push_back(sum_node);

                for (size_t k = 0; k < feature_dim; ++k) {
                    // 각 개별 곱셈 수행
                    size_t indexA = b * input_dim * feature_dim + i * feature_dim + k;
                    size_t indexB = k * output_dim + j;
                    double a_value = ptrA[indexA];
                    double b_value = ptrB[indexB];
                    double product = a_value * b_value;

                    // 각 곱셈 결과를 더하여 결과 행렬 값 계산
                    ptrResult[result_index] += product;

                    // 곱셈 노드 생성 및 덧셈 노드와 연결
                    std::shared_ptr<Node> mul_node = std::make_shared<Node>("multiply", a_value, b_value, product);
                    sum_node->add_child(mul_node);  // 곱셈 노드를 덧셈 노드의 자식으로 추가
                    mul_node->add_parent(sum_node); // 덧셈 노드를 곱셈 노드의 부모로 추가
                }

                // 덧셈 노드의 최종 출력 업데이트
                sum_node->output = ptrResult[result_index];
            }
        }
    }

    // 결과 행렬과 노드 리스트 반환
    return std::make_pair(result, node_list);
}
