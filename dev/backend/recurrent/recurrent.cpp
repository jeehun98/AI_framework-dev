#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <memory>
#include <cmath>
#include <string>
#include <unordered_map>
#include <iostream> // for logging
#include "../node/node.h"
#include "../activations/activations.h"
#include "../operaters/operations_matrix.h" // operations_matrix.h include

namespace py = pybind11;

std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> rnn_layer(
    py::array_t<double> input,
    py::array_t<double> weights,
    py::array_t<double> recurrent_weights,
    py::array_t<double> bias,
    const std::string& activation,
    std::vector<std::shared_ptr<Node>> node_list = {}
) {
    try {
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

        py::array_t<double> result({1, units});  // 마지막 타임스텝의 결과만 저장할 배열
        py::buffer_info bufResult = result.request();
        double* ptrInput = static_cast<double*>(bufInput.ptr);
        double* ptrResult = static_cast<double*>(bufResult.ptr);

        bool is_new_graph = node_list.empty();
        std::vector<double> state(units, 0.0);

        std::unordered_map<std::string, std::function<std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>>(
            py::array_t<double>, std::vector<std::shared_ptr<Node>>)>> activations_map;

        activations_map["relu"] = relu;
        activations_map["sigmoid"] = sigmoid;
        activations_map["tanh"] = tanh_activation;
        activations_map["leaky_relu"] = [alpha = 0.01](py::array_t<double> x, std::vector<std::shared_ptr<Node>> y) { return leaky_relu(x, alpha, y); };
        activations_map["softmax"] = softmax;


        if (activations_map.find(activation) == activations_map.end()) {
            throw std::runtime_error("Unsupported activation function: " + activation);
        }

        // Loop through each timestep
        for (int t = 0; t < timesteps; ++t) {
            py::array_t<double> input_at_t = py::array_t<double>(
                {1, input_dim},
                {input_dim * sizeof(double), sizeof(double)},
                ptrInput + t * input_dim
            );

            auto input_multiply_result = matrix_multiply(input_at_t, weights, node_list);
            auto& input_multiply = input_multiply_result.first;
            auto& input_multiply_node_list = input_multiply_result.second;

            py::array_t<double> state_arr = py::array_t<double>(
                {1, units},
                {units * sizeof(double), sizeof(double)},
                state.data()
            );

            auto recurrent_multiply_result = matrix_multiply(state_arr, recurrent_weights, node_list);
            auto& recurrent_multiply = recurrent_multiply_result.first;
            auto& recurrent_multiply_node_list = recurrent_multiply_result.second;

            if (input_multiply.ndim() == 1) {
                input_multiply = input_multiply.reshape(std::vector<py::ssize_t>{1, input_multiply.size()});
            }
            if (recurrent_multiply.ndim() == 1) {
                recurrent_multiply = recurrent_multiply.reshape(std::vector<py::ssize_t>{1, recurrent_multiply.size()});
            }

            auto sum_with_bias_result = matrix_add(input_multiply, recurrent_multiply, node_list);
            auto& sum_with_bias = sum_with_bias_result.first;
            auto& state_sum_node_list = sum_with_bias_result.second;

            if (sum_with_bias.ndim() == 1) {
                sum_with_bias = sum_with_bias.reshape(std::vector<py::ssize_t>{1, sum_with_bias.size()});
            }
            if (bias.ndim() == 1) {
                bias = bias.reshape(std::vector<py::ssize_t>{1, bias.size()});
            }

            auto output_with_bias_result = matrix_add(sum_with_bias, bias, node_list);
            auto& output_with_bias = output_with_bias_result.first;
            auto& output_bias_node_list = output_with_bias_result.second;

            auto activation_result = activations_map[activation](output_with_bias, node_list);

            for (int u = 0; u < units; ++u) {
                if (t == timesteps - 1) {
                    ptrResult[u] = activation_result.first.at(u);  // 마지막 타임스텝의 결과만 저장
                }
                state[u] = activation_result.first.at(u);
                auto& activation_node = activation_result.second[u];
                auto activation_leaf_node = activation_node->find_leaf_nodes()[0];

                auto state_sum_node = state_sum_node_list[u];
                auto input_multiply_node = input_multiply_node_list[u];
                auto recurrent_multiply_node = recurrent_multiply_node_list[u];

                state_sum_node->add_child(input_multiply_node);
                input_multiply_node->add_parent(state_sum_node);

                state_sum_node->add_child(recurrent_multiply_node);
                recurrent_multiply_node->add_parent(state_sum_node);

                auto output_bias_node = output_bias_node_list[u];
                output_bias_node->add_child(state_sum_node);
                state_sum_node->add_parent(output_bias_node);

                output_bias_node->add_parent(activation_leaf_node);
                activation_leaf_node->add_child(output_bias_node);

                if (t == timesteps - 1) {
                    node_list.push_back(activation_node);
                }
            }
        }

        return std::make_pair(result, node_list);
    } catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        throw;
    }
}
