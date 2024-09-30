#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <limits>
#include "../node/node.h"  // Node 클래스가 정의된 헤더 파일 포함

namespace py = pybind11;

std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> conv2d(
    py::array_t<double> input,  // 입력 이미지
    py::array_t<double> filters, // 필터 (커널)
    int stride = 1,
    std::string padding = "valid",
    std::vector<std::shared_ptr<Node>> node_list = {}
) {
    // 입력값과 커널 가중치 버퍼 정보 가져오기
    py::buffer_info bufInput = input.request();
    py::buffer_info bufFilters = filters.request();

    // 입력 및 필터 차원 체크
    if (bufInput.ndim != 3 || bufFilters.ndim != 4) {
        throw std::runtime_error("Input should be 3-D (height, width, channels) and Filters should be 4-D (filter_height, filter_width, in_channels, out_channels) NumPy arrays");
    }

    int input_height = bufInput.shape[0];
    int input_width = bufInput.shape[1];
    int input_channels = bufInput.shape[2];

    int filter_height = bufFilters.shape[1];
    int filter_width = bufFilters.shape[2];
    int filter_in_channels = bufFilters.shape[3];
    int filter_out_channels = bufFilters.shape[0];

    // 입력 채널 수와 필터의 입력 채널 수가 일치해야 함
    if (input_channels != filter_in_channels) {
        throw std::runtime_error("The number of input channels must match the number of filter input channels");
    }

    // Padding 설정
    int pad_height = 0;
    int pad_width = 0;

    if (padding == "same") {
        pad_height = (filter_height - 1) / 2;
        pad_width = (filter_width - 1) / 2;
    }

    // 출력 크기 계산
    int output_height = (input_height + 2 * pad_height - filter_height) / stride + 1;
    int output_width = (input_width + 2 * pad_width - filter_width) / stride + 1;

    // 결과 배열 생성
    py::array_t<double> result = py::array_t<double>({output_height, output_width, filter_out_channels});
    py::buffer_info bufResult = result.request();

    double* ptrInput = static_cast<double*>(bufInput.ptr);
    double* ptrFilters = static_cast<double*>(bufFilters.ptr);
    double* ptrResult = static_cast<double*>(bufResult.ptr);

    bool is_new_graph = node_list.empty();

    // Padding 적용된 입력 데이터 생성
    std::vector<double> padded_input((input_height + 2 * pad_height) * (input_width + 2 * pad_width) * input_channels, 0);
    for (int c = 0; c < input_channels; ++c) {
        for (int i = 0; i < input_height; ++i) {
            for (int j = 0; j < input_width; ++j) {
                int padded_index = ((i + pad_height) * (input_width + 2 * pad_width) + (j + pad_width)) * input_channels + c;
                int input_index = (i * input_width + j) * input_channels + c;
                padded_input[padded_index] = ptrInput[input_index];
            }
        }
    }

    // Convolution 연산
    for (int h = 0; h < output_height; ++h) {
        for (int w = 0; w < output_width; ++w) {
            for (int out_ch = 0; out_ch < filter_out_channels; ++out_ch) {
                int result_index = (h * output_width + w) * filter_out_channels + out_ch;
                ptrResult[result_index] = 0;  // 결과 초기화

                std::shared_ptr<Node> sum_node;

                if (is_new_graph) {
                    sum_node = std::make_shared<Node>("add", 0.0, 0.0, 0.0, 0.0);
                    node_list.push_back(sum_node);
                } else {
                    sum_node = node_list[result_index];
                    sum_node->output = 0.0;
                }

                for (int in_ch = 0; in_ch < input_channels; ++in_ch) {
                    for (int i = 0; i < filter_height; ++i) {
                        for (int j = 0; j < filter_width; ++j) {
                            int padded_i = h * stride + i;
                            int padded_j = w * stride + j;

                            int padded_index = ((padded_i * (input_width + 2 * pad_width)) + padded_j) * input_channels + in_ch;
                            int filter_index = ((i * filter_width + j) * input_channels + in_ch) * filter_out_channels + out_ch;

                            if (padded_index >= padded_input.size() || filter_index >= bufFilters.size) {
                                throw std::runtime_error("Index out of range");
                            }

                            double input_value = padded_input[padded_index];
                            double filter_value = ptrFilters[filter_index];

                            double weight = is_new_graph ? filter_value : sum_node->children[in_ch]->weight_value;  
                            double product = input_value * weight;

                            if (is_new_graph) {
                                std::shared_ptr<Node> mul_node = std::make_shared<Node>("multiply", input_value, weight, product, weight);
                                sum_node->add_child(mul_node);
                                mul_node->add_parent(sum_node);
                            } else {
                                auto mul_node = sum_node->get_children()[in_ch];
                                mul_node->update(input_value, weight, product, weight);
                            }

                            ptrResult[result_index] += product;
                        }
                    }
                }

                sum_node->output = ptrResult[result_index];
            }
        }
    }

    return std::make_pair(result, node_list);
}
