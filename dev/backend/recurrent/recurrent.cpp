#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <memory>
#include <cmath>
#include "../node/node.h"  // Node 클래스 포함

namespace py = pybind11;

std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> rnn_layer(
    py::array_t<double> input,  // 입력 시퀀스 데이터
    py::array_t<double> weights,  // 입력에 대한 가중치
    py::array_t<double> recurrent_weights,  // 상태에 대한 가중치
    py::array_t<double> bias,  // 바이어스
    std::string activation,  // 활성화 함수 ("tanh" 또는 "sigmoid" 등)
    std::vector<std::shared_ptr<Node>> node_list = {}
) {
    py::buffer_info bufInput = input.request();
    py::buffer_info bufWeights = weights.request();
    py::buffer_info bufRecurrentWeights = recurrent_weights.request();
    py::buffer_info bufBias = bias.request();

    int timesteps = bufInput.shape[0];
    int input_dim = bufInput.shape[1];
    int units = bufWeights.shape[1];

    if (bufWeights.ndim != 2 || bufRecurrentWeights.ndim != 2 || bufBias.ndim != 1) {
        throw std::runtime_error("Weights, recurrent weights, and bias must be 2-D, 2-D, and 1-D arrays, respectively.");
    }

    if (bufWeights.shape[0] != input_dim || bufRecurrentWeights.shape[0] != units || bufRecurrentWeights.shape[1] != units || bufBias.shape[0] != units) {
        throw std::runtime_error("Shape mismatch among inputs, weights, recurrent weights, or bias.");
    }

    py::array_t<double> result({timesteps, units});
    py::buffer_info bufResult = result.request();

    double* ptrInput = static_cast<double*>(bufInput.ptr);
    double* ptrWeights = static_cast<double*>(bufWeights.ptr);
    double* ptrRecurrentWeights = static_cast<double*>(bufRecurrentWeights.ptr);
    double* ptrBias = static_cast<double*>(bufBias.ptr);
    double* ptrResult = static_cast<double*>(bufResult.ptr);

    bool is_new_graph = node_list.empty();

    std::vector<double> state(units, 0.0);

    for (int t = 0; t < timesteps; ++t) {
        for (int u = 0; u < units; ++u) {
            std::shared_ptr<Node> input_sum_node;
            std::shared_ptr<Node> recurrent_sum_node;
            std::shared_ptr<Node> sum_node;

            if (is_new_graph) {
                input_sum_node = std::make_shared<Node>("add", 0.0, 0.0, 0.0, 0.0);
                recurrent_sum_node = std::make_shared<Node>("add", 0.0, 0.0, 0.0, 0.0);
                sum_node = std::make_shared<Node>("add", 0.0, 0.0, 0.0, 0.0);
            } else {
                input_sum_node = node_list[t * units + u]->get_children()[0]->get_children()[0];
                recurrent_sum_node = node_list[t * units + u]->get_children()[0]->get_children()[1];
                sum_node = node_list[t * units + u]->get_children()[0];
                input_sum_node->output = 0.0;
                recurrent_sum_node->output = 0.0;
                sum_node->output = 0.0;
            }

            // 입력 벡터와 가중치 곱의 결과를 input_sum_node에 추가하고 누적
            for (int i = 0; i < input_dim; ++i) {
                double input_value = ptrInput[t * input_dim + i];
                double weight = ptrWeights[i * units + u];
                double product = input_value * weight;

                std::shared_ptr<Node> mul_node;
                if (is_new_graph) {
                    mul_node = std::make_shared<Node>("multiply", input_value, weight, product, weight);
                    input_sum_node->add_child(mul_node);
                    mul_node->add_parent(input_sum_node);
                } else {
                    mul_node = input_sum_node->get_children()[i];
                    mul_node->update(input_value, weight, product, weight);
                }
                input_sum_node->output += product;
            }

            // 이전 은닉 상태와 순환 가중치 곱의 결과를 recurrent_sum_node에 추가하고 누적
            for (int h = 0; h < units; ++h) {
                double prev_state_value = state[h];
                double recurrent_weight = ptrRecurrentWeights[h * units + u];
                double product = prev_state_value * recurrent_weight;

                std::shared_ptr<Node> mul_node;
                if (is_new_graph) {
                    mul_node = std::make_shared<Node>("multiply", prev_state_value, recurrent_weight, product, recurrent_weight);
                    recurrent_sum_node->add_child(mul_node);
                    mul_node->add_parent(recurrent_sum_node);
                } else {
                    mul_node = recurrent_sum_node->get_children()[h];
                    mul_node->update(prev_state_value, recurrent_weight, product, recurrent_weight);
                }
                recurrent_sum_node->output += product;
            }

            // input_sum_node와 recurrent_sum_node를 sum_node에 연결하고 합산
            sum_node->output = input_sum_node->output + recurrent_sum_node->output;

            // 바이어스를 sum_node에 추가
            double bias_value = ptrBias[u];
            std::shared_ptr<Node> bias_node;
            if (is_new_graph) {
                bias_node = std::make_shared<Node>("bias", bias_value, 0.0, 0.0, 0.0);
                sum_node->add_parent(bias_node);
                bias_node->add_child(sum_node);
            } else {
                bias_node = sum_node->get_parents()[0];
                bias_node->update(bias_value, 0.0, 0.0, 0.0);
            }
            sum_node->output += bias_value;

            // 활성화 노드 생성 또는 업데이트
            std::shared_ptr<Node> activation_node;
            if (is_new_graph) {
                if (activation == "tanh") {
                    activation_node = std::make_shared<Node>("tanh", sum_node->output, std::tanh(sum_node->output), 0);
                } else if (activation == "sigmoid") {
                    activation_node = std::make_shared<Node>("sigmoid", sum_node->output, 1.0 / (1.0 + std::exp(-sum_node->output)), 0);
                }
                activation_node->add_child(bias_node);
                bias_node->add_parent(activation_node);
                node_list.push_back(activation_node);
            } else {
                activation_node = node_list[t * units + u];
                if (activation == "tanh") {
                    activation_node->update(sum_node->output, 0.0, std::tanh(sum_node->output), 0);
                } else if (activation == "sigmoid") {
                    activation_node->update(sum_node->output, 0.0, 1.0 / (1.0 + std::exp(-sum_node->output)), 0);
                }
            }

            ptrResult[t * units + u] = activation_node->output;
            state[u] = activation_node->output;
        }
    }

    return std::make_pair(result, node_list);
}
