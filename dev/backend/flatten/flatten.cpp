#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <memory>
#include "../node/node.h"  // Node 클래스가 정의된 헤더 파일 포함

namespace py = pybind11;

std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> flatten(
    py::array_t<double> input,  
    std::vector<std::shared_ptr<Node>> node_list = {}
) {
    py::buffer_info bufInput = input.request();
    
    // 1차원, 2차원, 3차원 입력에 따라 구분
    int flattened_size = 1;
    int output_rows = 1;
    int output_cols = bufInput.shape[0]; // 기본적으로 1차원일 경우, 2차원 (1, input_length)으로 변환
    
    if (bufInput.ndim == 1) {
        flattened_size = bufInput.shape[0];
    } else if (bufInput.ndim == 2) {
        output_rows = bufInput.shape[0];
        output_cols = bufInput.shape[1];
        flattened_size = output_rows * output_cols;
    } else if (bufInput.ndim == 3) {
        output_rows = 1; // 3차원은 모두 Flatten 하여 1차원 배열로 반환
        output_cols = bufInput.shape[0] * bufInput.shape[1] * bufInput.shape[2];
        flattened_size = output_cols;
    } else {
        throw std::runtime_error("Input should be 1-D, 2-D or 3-D NumPy array");
    }

    // Flatten된 배열 생성, 1차원 입력일 경우 (1, input_length)로 반환
    py::array_t<double> result = py::array_t<double>({output_rows, output_cols});
    py::buffer_info bufResult = result.request();

    double* ptrInput = static_cast<double*>(bufInput.ptr);
    double* ptrResult = static_cast<double*>(bufResult.ptr);

    bool is_new_graph = node_list.empty();

    // Flatten 또는 그대로 복사 수행
    for (int i = 0; i < flattened_size; ++i) {
        ptrResult[i] = ptrInput[i];

        std::shared_ptr<Node> flatten_node;
        if (is_new_graph) {
            flatten_node = std::make_shared<Node>("flatten", ptrInput[i], 0.0, ptrInput[i], 0.0);
            node_list.push_back(flatten_node);
        } else {
            flatten_node = node_list[i];
            flatten_node->update(ptrInput[i], 0.0, ptrInput[i], 0.0);
        }
    }

    return std::make_pair(result, node_list);
}
