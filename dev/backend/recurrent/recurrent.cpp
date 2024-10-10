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

    auto apply_activation = [](double x, const std::string& act) -> double {
        if (act == "tanh") {
            return std::tanh(x);
        } else if (act == "sigmoid") {
            return 1.0 / (1.0 + std::exp(-x));
        }
        return x;
    };

    for (int t = 0; t < timesteps; ++t) {
        for (int u = 0; u < units; ++u) {
            double linear_sum = 0.0;
            std::shared_ptr<Node> sum_node;

            if (is_new_graph) {
                sum_node = std::make_shared<Node>("add", 0.0, 0.0, 0.0, 0.0);
                node_list.push_back(sum_node);
            } else {
                sum_node = node_list[t * units + u];
                sum_node->output = 0.0;
            }

            for (int i = 0; i < input_dim; ++i) {
                double input_value = ptrInput[t * input_dim + i];
                double weight = ptrWeights[i * units + u];
                double product = input_value * weight;

                if (is_new_graph) {
                    std::shared_ptr<Node> mul_node = std::make_shared<Node>("multiply", input_value, weight, product, weight);
                    sum_node->add_child(mul_node);
                    mul_node->add_parent(sum_node);
                } else {
                    auto mul_node = sum_node->get_children()[i];
                    mul_node->update(input_value, weight, product, weight);
                }

                linear_sum += product;
            }

            for (int h = 0; h < units; ++h) {
                double prev_state_value = state[h];
                double recurrent_weight = ptrRecurrentWeights[h * units + u];
                double product = prev_state_value * recurrent_weight;

                if (is_new_graph) {
                    std::shared_ptr<Node> mul_node = std::make_shared<Node>("multiply", prev_state_value, recurrent_weight, product, recurrent_weight);
                    sum_node->add_child(mul_node);
                    mul_node->add_parent(sum_node);
                } else {
                    auto mul_node = sum_node->get_children()[input_dim + h];
                    mul_node->update(prev_state_value, recurrent_weight, product, recurrent_weight);
                }

                linear_sum += product;
            }

            linear_sum += ptrBias[u];
            double activated_output = apply_activation(linear_sum, activation);

            ptrResult[t * units + u] = activated_output;
            state[u] = activated_output;
            sum_node->output = activated_output;
        }
    }

    return std::make_pair(result, node_list);
}
