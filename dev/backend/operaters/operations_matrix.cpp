#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include "../node/node.h"  // Node 클래스가 정의된 헤더 파일 포함

namespace py = pybind11;

std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> matrix_add(
    py::array_t<double> A, 
    py::array_t<double> B,
    std::vector<std::shared_ptr<Node>> node_list = {}
) {
    // Numpy 배열의 버퍼 정보 가져오기
    py::buffer_info bufA = A.request(); 
    py::buffer_info bufB = B.request();

    // 입력 배열이 2D인지 확인 (행 크기 x 열 크기)
    if (bufA.ndim != 2 || bufB.ndim != 2) {
        throw std::runtime_error("Input should be 2-D NumPy arrays");
    }

    // 행과 열의 크기 확인
    if (bufA.shape[0] != bufB.shape[0] || bufA.shape[1] != bufB.shape[1]) {
        throw std::runtime_error("Input matrices must have the same shape");
    }

    size_t rows = bufA.shape[0];
    size_t cols = bufA.shape[1];

    // 결과 배열 생성
    py::array_t<double> result = py::array_t<double>({rows, cols});
    py::buffer_info bufResult = result.request();

    double* ptrA = static_cast<double*>(bufA.ptr);
    double* ptrB = static_cast<double*>(bufB.ptr);
    double* ptrResult = static_cast<double*>(bufResult.ptr);

    // 노드 리스트가 비어 있다면 새로운 노드 리스트 생성
    bool is_new_graph = node_list.empty();

    // 배치 단위로 행렬 덧셈 및 노드 생성
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            size_t index = i * cols + j;
            double valueA = ptrA[index];
            double valueB = ptrB[index];
            
            // 기존 노드가 있을 경우 해당 노드의 가중치 사용, 없으면 valueB 사용
            double weight = is_new_graph ? valueB : node_list[index]->weight_value;
            double sum = valueA + weight;

            // 덧셈 노드 생성
            if (is_new_graph) {
                std::shared_ptr<Node> add_node = std::make_shared<Node>("add", valueA, weight, sum, weight);
                node_list.push_back(add_node);
            } else {
                node_list[index]->update(valueA, weight, sum, weight);
            }

            // 결과 저장
            ptrResult[index] = sum;
        }
    }

    // 결과 배열과 노드 리스트 반환
    return std::make_pair(result, node_list);
}



// Dense 층의 순전파 연산에 사용되는 matrix_multiply,
// node_list 가 전달되지 않은 경우 : 첫 번째 연산에서 계산 그래프가 생성되면서 새로운 노드 리스트가 생성된다.
// node_list 가 전달된 경우 : 계산 그래프에 존재하는 가중치 값을 불러와 연산 수행

std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> matrix_multiply(
    py::array_t<double> A, 
    py::array_t<double> B, 
    std::vector<std::shared_ptr<Node>> node_list = {}
) {
    py::buffer_info bufA = A.request(), bufB = B.request();

    if (bufA.ndim != 2 || bufB.ndim != 2) {
        throw std::runtime_error("Input A and B should be 2-D NumPy arrays");
    }

    if (bufA.shape[1] != bufB.shape[0]) {
        throw std::runtime_error("The number of columns in A must match the number of rows in B");
    }

    size_t rows = bufA.shape[0];
    size_t feature_dim = bufA.shape[1];
    size_t cols = bufB.shape[1];

    py::array_t<double> result = py::array_t<double>({rows, cols});
    py::buffer_info bufResult = result.request();

    double* ptrA = static_cast<double*>(bufA.ptr);
    double* ptrB = static_cast<double*>(bufB.ptr);
    double* ptrResult = static_cast<double*>(bufResult.ptr);

    bool is_new_graph = node_list.empty();

    // 행렬 곱셈 수행
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            size_t result_index = i * cols + j;
            ptrResult[result_index] = 0;

            std::shared_ptr<Node> sum_node;
            if (is_new_graph) {
                sum_node = std::make_shared<Node>("add", 0.0, 0.0, 0.0, 0.0);
                node_list.push_back(sum_node);
            } else {
                sum_node = node_list[result_index];
                sum_node->output = 0.0;
            }

            for (size_t k = 0; k < feature_dim; ++k) {
                size_t indexA = i * feature_dim + k;
                size_t indexB = k * cols + j;

                double a_value = ptrA[indexA];
                double b_value = ptrB[indexB];

                double weight = is_new_graph ? b_value : sum_node->children[k]->weight_value;  // weight 가져오기
                double product = a_value * weight;

                // 곱셈 노드 생성 및 연결
                if (is_new_graph) {
                    std::shared_ptr<Node> mul_node = std::make_shared<Node>("multiply", a_value, weight, product, weight);
                    sum_node->add_child(mul_node);
                    mul_node->add_parent(sum_node);
                } else {
                    auto mul_node = sum_node->get_children()[k];
                    mul_node->update(a_value, weight, product, weight);
                }

                ptrResult[result_index] += product;
            }

            // 덧셈 노드의 최종 출력 업데이트
            sum_node->output = ptrResult[result_index];
        }
    }

    return std::make_pair(result, node_list);
}
