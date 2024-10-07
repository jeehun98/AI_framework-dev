#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <limits>
#include "../node/node.h"  // Node 클래스가 정의된 헤더 파일 포함

namespace py = pybind11;

std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> pooling2d(
    py::array_t<double> input,  // 입력 이미지
    int pool_height,            // 풀링 필터 높이
    int pool_width,             // 풀링 필터 너비
    std::pair<int, int> stride = {1, 1}, // 스트라이드: (stride_height, stride_width)
    std::string mode = "max",   // 풀링 모드: "max" 또는 "average"
    std::vector<std::shared_ptr<Node>> node_list = {}
) {
    // 입력값 버퍼 정보 가져오기
    py::buffer_info bufInput = input.request();

    // 입력 차원 체크
    if (bufInput.ndim != 3) {
        throw std::runtime_error("Input should be 3-D (height, width, channels) NumPy arrays");
    }

    int input_height = bufInput.shape[0];
    int input_width = bufInput.shape[1];
    int input_channels = bufInput.shape[2];

    // stride 튜플에서 각각의 stride 값 추출
    int stride_height = stride.first;
    int stride_width = stride.second;

    // 출력 크기 계산
    int output_height = (input_height - pool_height) / stride_height + 1;
    int output_width = (input_width - pool_width) / stride_width + 1;

    // 결과 배열 생성
    py::array_t<double> result = py::array_t<double>({output_height, output_width, input_channels});
    py::buffer_info bufResult = result.request();

    double* ptrInput = static_cast<double*>(bufInput.ptr);
    double* ptrResult = static_cast<double*>(bufResult.ptr);

    bool is_new_graph = node_list.empty();

    // 풀링 연산
    for (int h = 0; h < output_height; ++h) {
        for (int w = 0; w < output_width; ++w) {
            for (int ch = 0; ch < input_channels; ++ch) {
                int result_index = (h * output_width + w) * input_channels + ch;
                ptrResult[result_index] = (mode == "max") ? -std::numeric_limits<double>::infinity() : 0.0;

                std::shared_ptr<Node> pool_node;
                std::shared_ptr<Node> active_node = nullptr;

                // 각 풀링 영역에 대한 노드 설정
                if (is_new_graph) {
                    pool_node = std::make_shared<Node>((mode == "max") ? "max_pool" : "avg_pool", 0.0, 0.0, 0.0, 0.0);
                } else {
                    pool_node = node_list[result_index];
                    pool_node->output = (mode == "max") ? -std::numeric_limits<double>::infinity() : 0.0;
                }

                // 각 풀링 영역을 탐색하며 연산 수행
                for (int i = 0; i < pool_height; ++i) {
                    for (int j = 0; j < pool_width; ++j) {
                        int input_i = h * stride_height + i;
                        int input_j = w * stride_width + j;
                        int input_index = (input_i * input_width + input_j) * input_channels + ch;

                        double input_value = ptrInput[input_index];

                        if (mode == "max") {
                            // 최대값 찾기
                            if (input_value > ptrResult[result_index]) {
                                ptrResult[result_index] = input_value;

                                if (is_new_graph) {
                                    active_node = std::make_shared<Node>("max_pool", input_value, 0.0, input_value, 0.0);
                                } else {
                                    active_node = node_list[result_index];
                                    active_node->update(input_value, 0.0, input_value, 0.0);
                                }
                            }
                        } else if (mode == "average") {
                            // 평균값 계산
                            ptrResult[result_index] += input_value / (pool_height * pool_width);

                            if (is_new_graph) {
                                active_node = std::make_shared<Node>("avg_pool", input_value, 1.0 / (pool_height * pool_width), input_value / (pool_height * pool_width), 1.0 / (pool_height * pool_width));
                            } else {
                                active_node = node_list[result_index];
                                active_node->update(input_value, 1.0 / (pool_height * pool_width), input_value / (pool_height * pool_width), 1.0 / (pool_height * pool_width));
                            }
                        }
                    }
                }

                // active_node가 최종 부모 노드가 되고, pool_node는 자식 노드가 됨
                if (active_node != nullptr) {
                    if (is_new_graph) {
                        pool_node->add_child(active_node);  // pool_node가 자식으로 추가됨
                        active_node->add_parent(pool_node);
                    }
                    active_node->output = ptrResult[result_index];
                }

                // node_list에 최종 부모 노드를 추가
                if (is_new_graph) {
                    node_list.push_back(active_node);
                }
            }
        }
    }

    return std::make_pair(result, node_list);
}
