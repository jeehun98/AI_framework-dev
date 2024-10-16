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

        py::array_t<double> result({timesteps, units});
        py::buffer_info bufResult = result.request();

        double* ptrInput = static_cast<double*>(bufInput.ptr);
        double* ptrResult = static_cast<double*>(bufResult.ptr);

        bool is_new_graph = node_list.empty();
        std::vector<double> state(units, 0.0);

        auto activations_map = std::unordered_map<std::string, std::function<std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node> > >(
            py::array_t<double>, std::vector<std::shared_ptr<Node> >) > >{
            {"relu", relu},
            {"sigmoid", sigmoid},
            {"tanh", tanh_activation},
            {"leaky_relu", [alpha = 0.01](py::array_t<double> x, std::vector<std::shared_ptr<Node> > y) { return leaky_relu(x, alpha, y); }},
            {"softmax", softmax}
        };

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

            // 입력 가중치 간 연산 수행
            auto input_multiply_result = matrix_multiply(input_at_t, weights, node_list);

            // 직접 참조
            auto& input_multiply = input_multiply_result.first;

            // 길이는 unit 의 개수와 동일
            auto& input_multiply_node_list = input_multiply_result.second;

            // Recurrent multiplication
            py::array_t<double> state_arr = py::array_t<double>(
                {1, units},
                {units * sizeof(double), sizeof(double)},
                state.data()
            );

            // 순환 가중치 연산
            auto recurrent_multiply_result = matrix_multiply(state_arr, recurrent_weights, node_list);

            auto& recurrent_multiply = recurrent_multiply_result.first;
            
            // 길이는 unit 의 개수와 동일, 
            auto& recurrent_multiply_node_list = recurrent_multiply_result.second;

            // Sum with bias
            if (input_multiply.ndim() == 1) {
                input_multiply = input_multiply.reshape(std::vector<py::ssize_t>{1, input_multiply.size()});
            }
            if (recurrent_multiply.ndim() == 1) {
                recurrent_multiply = recurrent_multiply.reshape(std::vector<py::ssize_t>{1, recurrent_multiply.size()});
            }

            // bias 연산 전 두 input, recurrent 가중치 연산 결과 합치기
            auto sum_with_bias_result = matrix_add(input_multiply, recurrent_multiply, node_list);
            
            auto& sum_with_bias = sum_with_bias_result.first;
            auto& state_sum_node_list = sum_with_bias_result.second;

            // Final output with bias
            if (sum_with_bias.ndim() == 1) {
                sum_with_bias = sum_with_bias.reshape(std::vector<py::ssize_t>{1, sum_with_bias.size()});
            }
            if (bias.ndim() == 1) {
                bias = bias.reshape(std::vector<py::ssize_t>{1, bias.size()});
            }
            // bias 연산 수행
            auto output_with_bias_result = matrix_add(sum_with_bias, bias, node_list);
            auto& output_with_bias = output_with_bias_result.first;
            auto& output_bias_node_list = output_with_bias_result.second;

            auto activation_result = activations_map[activation](output_with_bias, node_list);

             // 연결하는 로직 구현하기
            // 각 연산 노드 간의 연결 설정
            for (size_t i = 0; i < units; ++i) {
                // 'state_sum_node_list'에서 현재 유닛의 노드 가져오기
                auto state_sum_node = state_sum_node_list[i];
                
                // 'input_multiply_node_list'에서 해당 유닛의 노드 가져오기
                auto input_multiply_node = input_multiply_node_list[i];
                // 'recurrent_multiply_node_list'에서 해당 유닛의 노드 가져오기
                auto recurrent_multiply_node = recurrent_multiply_node_list[i];

                // 'state_sum_node'에 'input_multiply_node'와 'recurrent_multiply_node' 추가
                state_sum_node->add_child(input_multiply_node);
                input_multiply_node->add_parent(state_sum_node);

                state_sum_node->add_child(recurrent_multiply_node);
                recurrent_multiply_node->add_parent(state_sum_node);

                auto output_bias_node = output_bias_node_list[i];

                output_bias_node->add_child(state_sum_node);
                state_sum_node->add_parent(output_bias_node);
            }

            for (int u = 0; u < units; ++u) {
                ptrResult[t * units + u] = activation_result.first.at(u);
                state[u] = activation_result.first.at(u);
                //node_list.insert(node_list.end(), activation_result.second.begin(), activation_result.second.end());
            }
            
            if (t == timesteps - 1){
                node_list = output_bias_node_list;
            }
        }

        return std::make_pair(result, node_list);
    } catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        throw;
    }
}
