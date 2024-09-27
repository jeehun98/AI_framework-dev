#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include "../node/node.h"  // Node 클래스가 정의된 헤더 파일 포함

namespace py = pybind11;
// 어떤 값을 전달하고 어디까지 컨트롤할지 패딩도 여기서 할거야?
std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> conv2d(
    py::array_t<double> input,  // 입력 이미지
    py::array_t<double> filters, // 필터 (커널)
    int stride = 1,
    std::string padding = "valid",
    std::vector<std::shared_ptr<Node>> node_list = {}
) {
    // 입력값과 커널 가중치
    py::buffer_info bufInput = input.request();
    py::buffer_info bufFilters = filters.request();

    // 각 입력의 형태가 이거 맞나
    // 각 feature map 하나씩만 사용할건지, 채널 전체 다 할 지
    if (bufInput.ndim != 3 || bufFilters.ndim != 4) {
        throw std::runtime_error("Input should be 3-D (height, width, channels) and Filters should be 4-D (filter_height, filter_width, in_channels, out_channels) NumPy arrays");
    }

    int input_height = bufInput.shape[0];
    int input_width = bufInput.shape[1];
    int input_channels = bufInput.shape[2];

    // 각 필터의 크기 (filter_height, filter_width, input_filter_dimension, output_filter_dimension)    
    int filter_height = bufFilters.shape[0];
    int filter_width = bufFilters.shape[1];
    // 입력 데이터 채널 수 
    int filter_in_channels = bufFilters.shape[2];
    int filter_out_channels = bufFilters.shape[3];

    // 그치 이건 맞지 입력 차원 별 각기 다른 필터가 존재해야 하므로
    if (input_channels != filter_in_channels) {
        throw std::runtime_error("The number of input channels must match the number of filter input channels");
    }

    // Padding 설정
    int pad_height = 0;
    int pad_width = 0;

    // Padding 조건식
    if (padding == "same") {
        pad_height = (filter_height - 1) / 2;
        pad_width = (filter_width - 1) / 2;
    }

    // 출력 형태 미리 계산
    int output_height = (input_height - filter_height + 2 * pad_height) / stride + 1;
    int output_width = (input_width - filter_width + 2 * pad_width) / stride + 1;

    // 계산한 result 형태의 선지정
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
    // 출력 index 에 맞게 데이터 삽입
    for (int h = 0; h < output_height; ++h) {
        for (int w = 0; w < output_width; ++w) {
            // 출력 차원의 개수만큼 반복
            for (int out_ch = 0; out_ch < filter_out_channels; ++out_ch) {
                // 출력 인덱스 1차원 리스트로 계산되므로 인덱스에 대한 계산이 필요하다. 
                int result_index = (h * output_width + w) * filter_out_channels + out_ch;
                // 메모리 주소로 접근
                // 초기화되어 있지 않은 result 값, 먼저 0으로 초기화
                ptrResult[result_index] = 0;

                // 노드 생성
                std::shared_ptr<Node> sum_node;

                // 계산 그래프 유무 체크
                if (is_new_graph) {
                    sum_node = std::make_shared<Node>("add", 0.0, 0.0, 0.0, 0.0);
                    node_list.push_back(sum_node);
                } else {
                    sum_node = node_list[result_index];
                    sum_node->output = 0.0;
                }

                // 입력 채널 반복
                for (int in_ch = 0; in_ch < input_channels; ++in_ch) {
                    
                    // 필터 높이 반복
                    for (int i = 0; i < filter_height; ++i) {

                        // 필터 너비 반복
                        for (int j = 0; j < filter_width; ++j) {

                            // 인덱스 계산하기... 
                            int padded_i = h * stride + i;
                            int padded_j = w * stride + j;

                            // 인덱스 계속 계산
                            int padded_index = ((padded_i * (input_width + 2 * pad_width)) + padded_j) * input_channels + in_ch;
                            int filter_index = ((i * filter_width + j) * input_channels + in_ch) * filter_out_channels + out_ch;

                            // 패딩과 채널별 연산되는 입력 데이터 값 지정
                            double input_value = padded_input[padded_index];
                            double filter_value = ptrFilters[filter_index];

                            
                            double weight = is_new_graph ? filter_value : sum_node->children[in_ch]->weight_value;  // weight 가져오기
                            double product = input_value * weight;

                            // 곱셈 노드 생성 및 연결
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

                // 덧셈 노드의 최종 출력 업데이트
                sum_node->output = ptrResult[result_index];
            }
        }
    }

    return std::make_pair(result, node_list);
}
