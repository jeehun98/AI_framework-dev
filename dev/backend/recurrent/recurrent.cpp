#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <memory>
#include <cmath>
#include <string>
#include <unordered_map>
#include "../node/node.h"
#include "../activations/activations.h"  // 활성화 함수 헤더 파일 포함

namespace py = pybind11;

std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> rnn_layer(
    py::array_t<double> input,  
    py::array_t<double> weights,  
    py::array_t<double> recurrent_weights,  
    py::array_t<double> bias,  
    const std::string& activation,  
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

    // 노드 리스트에는 그럼 어떤 값들이 저장되어 있는지에 대해 생각
    // activation 의 결과값, activation node 를 node_list 에 추가했었음, unit 의 개수임!
    bool is_new_graph = node_list.empty();
    std::vector<double> state(units, 0.0);

    // 활성화 함수 맵 설정
    std::unordered_map<std::string, std::function<std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>>(
        py::array_t<double>, std::vector<std::shared_ptr<Node>>)>> activations_map = {
        {"relu", relu},
        {"sigmoid", sigmoid},
        {"tanh", tanh_activation},
        {"leaky_relu", [alpha = 0.01](py::array_t<double> x, std::vector<std::shared_ptr<Node>> y) { return leaky_relu(x, alpha, y); }},
        {"softmax", softmax}
    };

    if (activations_map.find(activation) == activations_map.end()) {
        throw std::runtime_error("Unsupported activation function: " + activation);
    }

    for (int t = 0; t < timesteps; ++t) {
        for (int u = 0; u < units; ++u) {
            std::shared_ptr<Node> input_sum_node;
            std::shared_ptr<Node> recurrent_sum_node;
            std::shared_ptr<Node> sum_node;
            std::shared_ptr<Node> activation_node;

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

            sum_node->output = input_sum_node->output + recurrent_sum_node->output;

            // sum_node에 input_sum_node와 recurrent_sum_node를 연결
            sum_node->add_child(input_sum_node);
            sum_node->add_child(recurrent_sum_node);
            input_sum_node->add_parent(sum_node);
            recurrent_sum_node->add_parent(sum_node);

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

            py::array_t<double> sum_node_output({1}, &sum_node->output);
            std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> activation_result = activations_map[activation](sum_node_output, node_list);

            if (is_new_graph) {
                activation_node = activation_result.second.back();  // 새 노드 생성
                node_list.push_back(activation_node);

                auto leaf_nodes = activation_node->find_leaf_nodes();
                for (auto& leaf : leaf_nodes) {
                    leaf->add_child(bias_node);
                    bias_node->add_parent(leaf);
                }
            } else {
                activation_node = node_list[t * units + u];
                activation_node->update(sum_node->output, 0.0, activation_result.first.at(0), 0);

                auto leaf_nodes = activation_node->find_leaf_nodes();
                for (auto& leaf : leaf_nodes) {
                    leaf->add_child(bias_node);
                    bias_node->add_parent(leaf);
                }
            }

            ptrResult[t * units + u] = activation_result.first.at(0);
            state[u] = ptrResult[t * units + u];
        }
    }


    return std::make_pair(result, node_list);
}
