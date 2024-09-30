#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <limits>
#include "../node/node.h"  // Node 클래스가 정의된 헤더 파일 포함

namespace py = pybind11;

std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> max_pooling(
    py::array_t<double> input,  // 입력 이미지
    int pool_size = 2,          // 풀링 윈도우 크기 (2x2)
    int stride = 2,             // 풀링 stride 크기
    std::string padding = "valid",
    std::vector<std::shared_ptr<Node>> node_list = {}
) {
    // 입력 데이터 정보
    py::buffer_info bufInput = input.request();

    if (bufInput.ndim != 3) {
        throw std::runtime_error("Input should be 3-D (height, width, channels) NumPy arrays");
    }

    int input_height = bufInput.shape[0];
    int input_width = bufInput.shape[1];
    int input_channels = bufInput.shape[2];

    // Padding 설정
    int pad_height = 0;
    int pad_width = 0;

    if (padding == "same") {
        pad_height = (pool_size - 1) / 2;
        pad_width = (pool_size - 1) / 2;
    }

    // 출력 크기 계산
    int output_height = (input_height + 2 * pad_height - pool_size) / stride + 1;
    int output_width = (input_width + 2 * pad_width - pool_size) / stride + 1;

    // 결과 배열 생성
    py::array_t<double> result = py::array_t<double>({output_height, output_width, input_channels});
    py::buffer_info bufResult = result.request();

    double* ptrInput = static_cast<double*>(bufInput.ptr);
    double* ptrResult = static_cast<double*>(bufResult.ptr);

    bool is_new_graph = node_list.empty();

    // Pooling 연산
    for (int h = 0; h < output_height; ++h) {
        for (int w = 0; w < output_width; ++w) {
            for (int ch = 0; ch < input_channels; ++ch) {
                int result_index = (h * output_width + w) * input_channels + ch;
                double max_value = -std::numeric_limits<double>::infinity();

                std::shared_ptr<Node> max_node;
                if (is_new_graph) {
                    max_node = std::make_shared<Node>("max", 0.0, 0.0, 0.0, 0.0);
                    node_list.push_back(max_node);
                } else {
                    max_node = node_list[result_index];
                    max_node->output = -std::numeric_limits<double>::infinity();
                }

                for (int i = 0; i < pool_size; ++i) {
                    for (int j = 0; j < pool_size; ++j) {
                        int input_i = h * stride + i - pad_height;
                        int input_j = w * stride + j - pad_width;

                        if (input_i >= 0 && input_i < input_height && input_j >= 0 && input_j < input_width) {
                            int input_index = (input_i * input_width + input_j) * input_channels + ch;
                            double input_value = ptrInput[input_index];

                            if (input_value > max_value) {
                                max_value = input_value;
                                if (is_new_graph) {
                                    max_node->update(input_value, 1.0, max_value, 1.0);
                                } else {
                                    max_node->update(input_value, max_node->weight_value, max_value, max_node->weight_value);
                                }
                            }
                        }
                    }
                }

                ptrResult[result_index] = max_value;
                max_node->output = max_value;
            }
        }
    }

    return std::make_pair(result, node_list);
}
