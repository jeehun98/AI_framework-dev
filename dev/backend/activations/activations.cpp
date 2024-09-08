#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include "../node/node.h"  // Node 클래스가 정의된 헤더 파일 포함

namespace py = pybind11;

std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> relu(py::array_t<double> inputs) {
    py::buffer_info buf = inputs.request();
    double* ptr = static_cast<double*>(buf.ptr);

    py::array_t<double> result(buf.size);

    py::buffer_info buf_result = result.request();
    double* ptr_result = static_cast<double*>(buf_result.ptr);

    std::vector<std::shared_ptr<Node>> node_list;

    for (size_t i = 0; i < buf.size; ++i) {
        double input_value = ptr[i];
        double zero_value = 0.0;

        // 비교 노드 생성 (x > 0)
        std::shared_ptr<Node> compare_node = std::make_shared<Node>("compare", input_value, zero_value, input_value > 0 ? 1.0 : 0.0);

        // 선택 노드 생성 (x 또는 0 선택)
        double output_value = (compare_node->output > 0) ? input_value : 0.0;
        std::shared_ptr<Node> select_node = std::make_shared<Node>("select", input_value, zero_value, output_value);

        // 노드 연결
        select_node->add_child(compare_node);
        compare_node->add_parent(select_node);

        ptr_result[i] = output_value;

        node_list.push_back(select_node);
    }

    return std::make_pair(result, node_list);
}


// Sigmoid 연산을 개별 노드로 분리하여 구현
std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> sigmoid(py::array_t<double> inputs) {
    // Numpy 배열의 버퍼 정보 가져오기
    py::buffer_info buf = inputs.request();
    double* ptr = static_cast<double*>(buf.ptr);

    // 결과 배열 생성
    py::array_t<double> result(buf.size);
    py::buffer_info buf_result = result.request();
    double* ptr_result = static_cast<double*>(buf_result.ptr);

    // 노드 리스트 생성
    std::vector<std::shared_ptr<Node>> node_list;

    // 각 요소에 대해 Sigmoid 연산을 수행하고, 노드 생성
    for (size_t i = 0; i < buf.size; ++i) {
        double input_value = ptr[i];

        // Negate 노드 (-x)
        double neg_output = -input_value;
        std::shared_ptr<Node> neg_node = std::make_shared<Node>("negate", input_value, neg_output);

        // Exponentiate 노드 (exp(-x))
        double exp_output = std::exp(neg_output);
        std::shared_ptr<Node> exp_node = std::make_shared<Node>("exp", neg_output, exp_output);
        exp_node->add_child(neg_node);
        neg_node->add_parent(exp_node);

        // Add 1 노드 (1 + exp(-x))
        // 상수 1 까지의 연산, add 하는 객체가 2개이므로
        double constant_value = 1.0;
        double add_output = constant_value + exp_output;
        std::shared_ptr<Node> add_node = std::make_shared<Node>("add", exp_output, constant_value, add_output);

        // 부모-자식 관계 설정
        add_node->add_child(exp_node);  // exp_node는 add_node의 자식 노드
        exp_node->add_parent(add_node);   // add_node는 exp_node의 부모 노드

        // Reciprocal 노드 (1 / (1 + exp(-x)))
        double constant_value = 1.0;
        double recip_output = constant_value / add_output;  // 1.0을 분자에 포함
        std::shared_ptr<Node> recip_node = std::make_shared<Node>("reciprocal", constant_value, add_output, recip_output);

        recip_node->add_child(add_node);
        add_node->add_parent(recip_node);

        // 결과 저장
        ptr_result[i] = recip_output;

        // 노드 리스트에 추가
        node_list.push_back(recip_node);
    }

    // 결과 배열과 노드 리스트 반환
    return std::make_pair(result, node_list);
}

std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> tanh_activation(py::array_t<double> inputs) {
    py::buffer_info buf = inputs.request();
    double* ptr = static_cast<double*>(buf.ptr);

    py::array_t<double> result(buf.size);
    py::buffer_info buf_result = result.request();
    double* ptr_result = static_cast<double*>(buf_result.ptr);

    std::vector<std::shared_ptr<Node>> node_list;

    for (size_t i = 0; i < buf.size; ++i) {
        double input_value = ptr[i];

        // Exponentiate 노드 (exp(x))
        double exp_pos_output = std::exp(input_value);
        std::shared_ptr<Node> exp_pos_node = std::make_shared<Node>("exp", input_value, exp_pos_output);

        // Exponentiate 노드 (exp(-x))
        double exp_neg_output = std::exp(-input_value);
        std::shared_ptr<Node> exp_neg_node = std::make_shared<Node>("exp", -input_value, exp_neg_output);

        // Numerator 노드 (exp(x) - exp(-x))
        double numerator_output = exp_pos_output - exp_neg_output;
        std::shared_ptr<Node> numerator_node = std::make_shared<Node>("subtract", exp_pos_output, exp_neg_output, numerator_output);
        numerator_node->add_child(exp_pos_node);
        numerator_node->add_child(exp_neg_node);

        exp_pos_node->add_parent(numerator_node);
        exp_neg_node->add_parent(numerator_node);

        // Denominator 노드 (exp(x) + exp(-x))
        double denominator_output = exp_pos_output + exp_neg_output;
        std::shared_ptr<Node> denominator_node = std::make_shared<Node>("add", exp_pos_output, exp_neg_output, denominator_output);
        denominator_node->add_child(exp_pos_node);
        denominator_node->add_child(exp_neg_node);

        exp_pos_node->add_parent(denominator_node);
        exp_neg_node->add_parent(denominator_node);

        // Reciprocal 노드 (1 / (exp(x) + exp(-x)))
        double constant_value = 1.0;
        double reciprocal_output = constant_value / denominator_output;
        std::shared_ptr<Node> reciprocal_node = std::make_shared<Node>("reciprocal", constant_value, denominator_output, reciprocal_output);  // 두 입력: 상수 1과 분모
        reciprocal_node->add_child(denominator_node);
        denominator_node->add_parent(reciprocal_node);

        // Tanh 노드 (Numerator * Reciprocal)
        double tanh_output = numerator_output * reciprocal_output;
        std::shared_ptr<Node> tanh_node = std::make_shared<Node>("multiply", numerator_output, reciprocal_output, tanh_output);
        tanh_node->add_child(numerator_node);
        numerator_node->add_parent(tanh_node);

        ptr_result[i] = tanh_output;

        node_list.push_back(tanh_node);
    }

    return std::make_pair(result, node_list);
}


std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> leaky_relu(py::array_t<double> inputs, double alpha = 0.01) {
    // Numpy 배열의 버퍼 정보 가져오기
    py::buffer_info buf = inputs.request();
    double* ptr = static_cast<double*>(buf.ptr);

    // 결과 배열 생성
    py::array_t<double> result(buf.size);
    py::buffer_info buf_result = result.request();
    double* ptr_result = static_cast<double*>(buf_result.ptr);

    // 노드 리스트 생성 (마지막 노드만 추가)
    std::vector<std::shared_ptr<Node>> node_list;

    // 각 요소에 대해 Leaky ReLU 연산을 수행하고, 노드 생성
    for (size_t i = 0; i < buf.size; ++i) {
        double input_value = ptr[i];
        double leaky_value = alpha * input_value;

        // 비교 노드 생성 (x > 0)
        double compare_output = (input_value > 0) ? 1.0 : 0.0;
        std::shared_ptr<Node> compare_node = std::make_shared<Node>("compare", input_value, compare_output);

        // 선택 노드 생성 (x 또는 alpha * x 선택)
        double output_value = (compare_output > 0) ? input_value : leaky_value;
        std::shared_ptr<Node> select_node = std::make_shared<Node>("select", input_value, leaky_value, output_value);

        // 노드 연결: 선택 노드를 비교 노드의 자식으로 설정
        select_node->add_parent(compare_node);
        compare_node->add_child(select_node);

        // 결과 저장
        ptr_result[i] = output_value;

        // 노드 리스트에 마지막 노드(선택 노드)만 추가
        node_list.push_back(select_node);
    }

    // 결과 배열과 노드 리스트 반환
    return std::make_pair(result, node_list);
}


// Softmax 연산을 배치 데이터로 수행하도록 개별 노드로 분리하여 구현
std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> softmax(py::array_t<double> inputs) {
    py::buffer_info buf = inputs.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Input should be a 2-D array");

    size_t num_rows = buf.shape[0];
    size_t num_cols = buf.shape[1];
    double* ptr = static_cast<double*>(buf.ptr);

    py::array_t<double> result({num_rows, num_cols});
    py::buffer_info buf_result = result.request();
    double* ptr_result = static_cast<double*>(buf_result.ptr);

    std::vector<std::shared_ptr<Node>> node_list;

    // 각 행에 대해 Softmax 연산 수행
    for (size_t row = 0; row < num_rows; ++row) {
        double* row_ptr = ptr + row * num_cols;
        double* row_result_ptr = ptr_result + row * num_cols;

        // 각 행 별 최댓값
        double max_val = *std::max_element(row_ptr, row_ptr + num_cols);

        double sum = 0.0;
        std::vector<std::shared_ptr<Node>> exp_nodes;

        for (size_t i = 0; i < num_cols; ++i) {
            double input_value = row_ptr[i];

            // Subtract 노드 생성 (각 입력에서 최대값을 뺌)
            double sub_output = input_value - max_val;
            std::shared_ptr<Node> sub_node = std::make_shared<Node>("subtract", input_value, max_val, sub_output);

            // Exponentiate 노드 생성 (exp(x))
            double exp_output = std::exp(sub_output);
            std::shared_ptr<Node> exp_node = std::make_shared<Node>("exp", sub_output, exp_output);
            exp_node->add_child(sub_node);
            sub_node->add_parent(exp_node);

            exp_nodes.push_back(exp_node);

            sum += exp_output;
        }

        for (size_t i = 0; i < num_cols; ++i) {
            // Divide 노드 생성 (exp(x) / sum)
            double div_output = exp_nodes[i]->output / sum;
            std::shared_ptr<Node> div_node = std::make_shared<Node>("divide", exp_nodes[i]->output, sum, div_output);
            
            div_node->add_child(exp_nodes[i]);
            exp_nodes[i]->add_parent(div_node);

            // 결과의 개수는 입력 데이터의 개수와 동일, 배치까지
            node_list.push_back(div_node);

            // 결과 저장
            row_result_ptr[i] = div_output;
        }
    }

    return std::make_pair(result, node_list);
}