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

        // 입력 데이터의 각 타임 스텝, 길이 별로 반복 수행
        for (int t = 0; t < timesteps; ++t) {
            // input_at_t 에 대한 데이터의 이해
            // input 배열에서 t 번째 타임스텝의 데이터를 배열로 생성
            // 배열의 모양을 지정하는 부분, (1, input_dim) - 행 벡터
            // input 의 t 번째 타임스텝 데이터를 가리키는 시작 주소 지정
            // 메모리 주소에 대한 연산
            // 미리 지정된 크기
            py::array_t<double> input_at_t({1, input_dim}, ptrInput + t * input_dim);

            // Matrix multiplication for input
            // 입력 데이터의 연산
            // 초기의 경우 비어있는 node_list 가 전달되지만,
            // 계산 그래프가 존재하는 경우 node_list 를 어떻게 전달해줄지에 대한 고민
            // 적절한 연산으로 node_list 의 인덱스에 접근해야 한다.
            auto input_multiply_result = matrix_multiply(input_at_t, weights, node_list);
            auto& input_multiply = input_multiply_result.first;
            node_list = input_multiply_result.second;

            // Update for recurrent multiply
            py::array_t<double> state_arr({1, units}, state.data());
            auto recurrent_multiply_result = matrix_multiply(state_arr, recurrent_weights, node_list);
            auto& recurrent_multiply = recurrent_multiply_result.first;
            node_list = recurrent_multiply_result.second;

            // Update for sum with bias
            auto sum_with_bias_result = matrix_add(input_multiply, recurrent_multiply, node_list);
            auto& sum_with_bias = sum_with_bias_result.first;
            node_list = sum_with_bias_result.second;

            // Update for output with bias
            auto output_with_bias_result = matrix_add(sum_with_bias, bias, node_list);
            auto& output_with_bias = output_with_bias_result.first;
            node_list = output_with_bias_result.second;

            auto activation_result = activations_map[activation](output_with_bias, node_list);

            for (int u = 0; u < units; ++u) {
                ptrResult[t * units + u] = activation_result.first.at(u);
                state[u] = activation_result.first.at(u);
            }
        }

        return std::make_pair(result, node_list);
    } catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        throw;
    }
}
