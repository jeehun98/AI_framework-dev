#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include "../node/node.h"  // Node 클래스가 정의된 헤더 파일 포함

namespace py = pybind11;

// matrix_add 함수
py::array_t<double> matrix_add(py::array_t<double> A, py::array_t<double> B) {
    // Numpy 배열의 버퍼 정보 가져오기
    py::buffer_info bufA = A.request(), bufB = B.request();

    // 입력 배열이 2D인지 확인
    if (bufA.ndim != 2 || bufB.ndim != 2) {
        throw std::runtime_error("Input should be 2-D NumPy arrays");
    }

    // 행과 열의 크기 확인
    if (bufA.shape[0] != bufB.shape[0] || bufA.shape[1] != bufB.shape[1]) {
        throw std::runtime_error("Input matrices must have the same shape");
    }

    // 행과 열의 크기
    size_t rows = bufA.shape[0];
    size_t cols = bufA.shape[1];

    // 결과 배열 생성
    py::array_t<double> result = py::array_t<double>({rows, cols});
    py::buffer_info bufResult = result.request();

    double* ptrA = static_cast<double*>(bufA.ptr);
    double* ptrB = static_cast<double*>(bufB.ptr);
    double* ptrResult = static_cast<double*>(bufResult.ptr);

    // 행렬 덧셈
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            ptrResult[i * cols + j] = ptrA[i * cols + j] + ptrB[i * cols + j];
        }
    }

    return result;
}

// matrix_multiply 함수
std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> matrix_multiply(py::array_t<double> A, py::array_t<double> B, std::vector<std::shared_ptr<Node>>* node_list = nullptr) {
    // Numpy 배열의 버퍼 정보 가져오기
    py::buffer_info bufA = A.request(), bufB = B.request();

    // 입력 배열이 2D인지 확인
    if (bufA.ndim != 2 || bufB.ndim != 2) {
        throw std::runtime_error("Input should be 2-D NumPy arrays");
    }

    // 행렬 곱셈이 가능한지 확인
    if (bufA.shape[1] != bufB.shape[0]) {
        throw std::runtime_error("Inner matrix dimensions must agree");
    }

    // 행과 열의 크기
    size_t rows = bufA.shape[0];
    size_t cols = bufB.shape[1];
    size_t inner_dim = bufA.shape[1];

    // 결과 배열 생성
    py::array_t<double> result = py::array_t<double>({rows, cols});
    py::buffer_info bufResult = result.request();

    double* ptrA = static_cast<double*>(bufA.ptr);
    double* ptrB = static_cast<double*>(bufB.ptr);
    double* ptrResult = static_cast<double*>(bufResult.ptr);

    // 노드 리스트가 없으면 생성
    std::vector<std::shared_ptr<Node>> local_node_list;
    if (node_list == nullptr) {
        node_list = &local_node_list;
    }

    // 행렬 곱셈
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            ptrResult[i * cols + j] = 0;

            // 덧셈 노드 생성
            std::shared_ptr<Node> sum_node = std::make_shared<Node>("sum", 0, 0, 0.0);
            node_list->push_back(sum_node);

            for (size_t k = 0; k < inner_dim; ++k) {
                // 각 개별 곱셈 수행
                double a_value = ptrA[i * inner_dim + k];
                double b_value = ptrB[k * cols + j];
                double product = a_value * b_value;

                // 각 곱셈 결과를 더하여 결과 행렬 값 계산
                ptrResult[i * cols + j] += product;

                // 곱셈 노드 생성 및 덧셈 노드와 연결
                std::shared_ptr<Node> mul_node = std::make_shared<Node>("multiply", a_value, b_value, product);
                sum_node->add_child(mul_node);  // 곱셈 노드를 부모로 추가
                mul_node->add_parent(sum_node);   // 덧셈 노드를 자식으로 추가
            }

            // 덧셈 노드의 최종 출력 업데이트
            sum_node->output = ptrResult[i * cols + j];
        }
    }

    // 결과 행렬과 노드 리스트 반환
    return std::make_pair(result, *node_list);
}
