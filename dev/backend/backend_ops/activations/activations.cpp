#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include "../node/node.h"  // Node 클래스가 정의된 헤더 파일 포함

namespace py = pybind11;

std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> relu(
    py::array_t<double> inputs, 
    std::vector<std::shared_ptr<Node>> node_list = {}
) {
    py::buffer_info buf = inputs.request();
    double* ptr = static_cast<double*>(buf.ptr);

    py::array_t<double> result(buf.size);
    py::buffer_info buf_result = result.request();
    double* ptr_result = static_cast<double*>(buf_result.ptr);

    bool is_new_graph = node_list.empty();

    for (size_t i = 0; i < buf.size; ++i) {
        double input_value = ptr[i];
        double zero_value = 0.0;

        if (is_new_graph) {
            std::shared_ptr<Node> compare_node = std::make_shared<Node>("compare", input_value, zero_value, input_value > 0 ? 1.0 : 0.0);
            double output_value = (compare_node->output > 0) ? input_value : 0.0;
            std::shared_ptr<Node> select_node = std::make_shared<Node>("select", input_value, zero_value, output_value);

            select_node->add_child(compare_node);
            compare_node->add_parent(select_node);

            node_list.push_back(select_node);
            ptr_result[i] = output_value;
        } else {
            auto select_node = node_list[i];
            select_node->update(input_value, zero_value, (input_value > 0) ? input_value : 0.0, 0);
            ptr_result[i] = select_node->output;

            // 노드 연결 확인 및 재설정
            auto compare_node = select_node->get_children()[0];  // 첫 번째 자식 노드가 비교 노드
            compare_node->update(input_value, zero_value, input_value > 0 ? 1.0 : 0.0, 0);
        }
    }

    return std::make_pair(result, node_list);
}

std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> sigmoid(
    py::array_t<double> inputs, 
    std::vector<std::shared_ptr<Node>> node_list = {}
) {
    py::buffer_info buf = inputs.request();
    double* ptr = static_cast<double*>(buf.ptr);

    py::array_t<double> result(buf.size);
    py::buffer_info buf_result = result.request();
    double* ptr_result = static_cast<double*>(buf_result.ptr);

    bool is_new_graph = node_list.empty();

    for (size_t i = 0; i < buf.size; ++i) {
        double input_value = ptr[i];
        if (is_new_graph) {
            double neg_output = -input_value;
            std::shared_ptr<Node> neg_node = std::make_shared<Node>("negate", input_value, neg_output, 0);

            double exp_output = std::exp(neg_output);
            std::shared_ptr<Node> exp_node = std::make_shared<Node>("exp", neg_output, exp_output, 0);
            exp_node->add_child(neg_node);
            neg_node->add_parent(exp_node);

            double constant_value = 1.0;
            double add_output = constant_value + exp_output;
            std::shared_ptr<Node> add_node = std::make_shared<Node>("add", exp_output, constant_value, add_output, 0);
            add_node->add_child(exp_node);
            exp_node->add_parent(add_node);

            double recip_output = constant_value / add_output;
            std::shared_ptr<Node> recip_node = std::make_shared<Node>("reciprocal", constant_value, add_output, recip_output, 0);
            recip_node->add_child(add_node);
            add_node->add_parent(recip_node);

            node_list.push_back(recip_node);
            ptr_result[i] = recip_output;
        } else {
            auto recip_node = node_list[i];
            double neg_output = -input_value;
            double exp_output = std::exp(neg_output);
            double add_output = 1.0 + exp_output;
            recip_node->update(1.0, add_output, 1.0 / add_output, 0);
            ptr_result[i] = recip_node->output;

            // 노드 연결 확인 및 재설정
            auto add_node = recip_node->get_children()[0];
            add_node->update(exp_output, 1.0, add_output, 0);

            auto exp_node = add_node->get_children()[0];
            exp_node->update(neg_output, 0.0, exp_output, 0);

            auto neg_node = exp_node->get_children()[0];
            neg_node->update(input_value, 0.0, neg_output, 0);
        }
    }

    return std::make_pair(result, node_list);
}


std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> tanh_activation(
    py::array_t<double> inputs, 
    std::vector<std::shared_ptr<Node>> node_list = {}
) {
    py::buffer_info buf = inputs.request();
    double* ptr = static_cast<double*>(buf.ptr);

    py::array_t<double> result(buf.size);
    py::buffer_info buf_result = result.request();
    double* ptr_result = static_cast<double*>(buf_result.ptr);

    bool is_new_graph = node_list.empty();

    for (size_t i = 0; i < buf.size; ++i) {
        double input_value = ptr[i];

        if (is_new_graph) {
            double exp_pos_output = std::exp(input_value);
            std::shared_ptr<Node> exp_pos_node = std::make_shared<Node>("exp", input_value, exp_pos_output, 0);

            double exp_neg_output = std::exp(-input_value);
            std::shared_ptr<Node> exp_neg_node = std::make_shared<Node>("exp", -input_value, exp_neg_output, 0);

            double numerator_output = exp_pos_output - exp_neg_output;
            std::shared_ptr<Node> numerator_node = std::make_shared<Node>("subtract", exp_pos_output, exp_neg_output, numerator_output, 0);
            numerator_node->add_child(exp_pos_node);
            numerator_node->add_child(exp_neg_node);
            exp_pos_node->add_parent(numerator_node);
            exp_neg_node->add_parent(numerator_node);

            double denominator_output = exp_pos_output + exp_neg_output;
            std::shared_ptr<Node> denominator_node = std::make_shared<Node>("add", exp_pos_output, exp_neg_output, denominator_output, 0);
            denominator_node->add_child(exp_pos_node);
            denominator_node->add_child(exp_neg_node);
            exp_pos_node->add_parent(denominator_node);
            exp_neg_node->add_parent(denominator_node);

            double reciprocal_output = 1.0 / denominator_output;
            std::shared_ptr<Node> reciprocal_node = std::make_shared<Node>("reciprocal", 1.0, denominator_output, reciprocal_output, 0);
            reciprocal_node->add_child(denominator_node);
            denominator_node->add_parent(reciprocal_node);

            double tanh_output = numerator_output * reciprocal_output;
            std::shared_ptr<Node> tanh_node = std::make_shared<Node>("multiply", numerator_output, reciprocal_output, tanh_output, reciprocal_output);
            tanh_node->add_child(numerator_node);
            numerator_node->add_parent(tanh_node);

            node_list.push_back(tanh_node);
            ptr_result[i] = tanh_output;
        } else {
            auto tanh_node = node_list[i];
            double exp_pos_output = std::exp(input_value);
            double exp_neg_output = std::exp(-input_value);
            double numerator_output = exp_pos_output - exp_neg_output;
            double denominator_output = exp_pos_output + exp_neg_output;
            double tanh_output = numerator_output / denominator_output;
            tanh_node->update(numerator_output, denominator_output, tanh_output, denominator_output);
            ptr_result[i] = tanh_node->output;

            // 노드 연결 확인 및 재설정
            auto reciprocal_node = tanh_node->get_parents()[0];
            reciprocal_node->update(1.0, denominator_output, 1.0 / denominator_output, 0);

            auto denominator_node = reciprocal_node->get_parents()[0];
            denominator_node->update(exp_pos_output, exp_neg_output, denominator_output, 0);

            auto numerator_node = tanh_node->get_parents()[0];
            numerator_node->update(exp_pos_output, exp_neg_output, numerator_output, 0);
        }
    }

    return std::make_pair(result, node_list);
}

std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> leaky_relu(
    py::array_t<double> inputs, 
    double alpha = 0.01, 
    std::vector<std::shared_ptr<Node>> node_list = {}
) {
    py::buffer_info buf = inputs.request();
    double* ptr = static_cast<double*>(buf.ptr);

    py::array_t<double> result(buf.size);
    py::buffer_info buf_result = result.request();
    double* ptr_result = static_cast<double*>(buf_result.ptr);

    bool is_new_graph = node_list.empty();

    for (size_t i = 0; i < buf.size; ++i) {
        double input_value = ptr[i];
        double leaky_value = alpha * input_value;

        if (is_new_graph) {
            double compare_output = (input_value > 0) ? 1.0 : 0.0;
            std::shared_ptr<Node> compare_node = std::make_shared<Node>("compare", input_value, compare_output, 0);

            double output_value = (compare_output > 0) ? input_value : leaky_value;
            std::shared_ptr<Node> select_node = std::make_shared<Node>("select", input_value, leaky_value, output_value, 0);
            select_node->add_parent(compare_node);
            compare_node->add_child(select_node);

            node_list.push_back(select_node);
            ptr_result[i] = output_value;
        } else {
            auto select_node = node_list[i];
            double output_value = (input_value > 0) ? input_value : leaky_value;
            select_node->update(input_value, leaky_value, output_value, 0);
            ptr_result[i] = select_node->output;

            // 노드 연결 확인 및 재설정
            auto compare_node = select_node->get_parents()[0];
            compare_node->update(input_value, 0.0, input_value > 0 ? 1.0 : 0.0, 0);
        }
    }

    return std::make_pair(result, node_list);
}

std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> softmax(
    py::array_t<double> inputs, 
    std::vector<std::shared_ptr<Node>> node_list = {}
) {
    py::buffer_info buf = inputs.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Input should be a 2-D array");

    size_t num_rows = buf.shape[0];
    size_t num_cols = buf.shape[1];
    double* ptr = static_cast<double*>(buf.ptr);

    py::array_t<double> result({num_rows, num_cols});
    py::buffer_info buf_result = result.request();
    double* ptr_result = static_cast<double*>(buf_result.ptr);

    bool is_new_graph = node_list.empty();

    for (size_t row = 0; row < num_rows; ++row) {
        double* row_ptr = ptr + row * num_cols;
        double* row_result_ptr = ptr_result + row * num_cols;

        double max_val = *std::max_element(row_ptr, row_ptr + num_cols);

        double sum = 0.0;
        std::vector<std::shared_ptr<Node>> exp_nodes;

        for (size_t i = 0; i < num_cols; ++i) {
            double input_value = row_ptr[i];

            if (is_new_graph) {
                double sub_output = input_value - max_val;
                std::shared_ptr<Node> sub_node = std::make_shared<Node>("subtract", input_value, max_val, sub_output, 0);

                double exp_output = std::exp(sub_output);
                std::shared_ptr<Node> exp_node = std::make_shared<Node>("exp", sub_output, exp_output, 0);
                exp_node->add_child(sub_node);
                sub_node->add_parent(exp_node);

                exp_nodes.push_back(exp_node);
                sum += exp_output;
            } else {
                auto exp_node = node_list[row * num_cols + i];
                double sub_output = input_value - max_val;
                double exp_output = std::exp(sub_output);
                exp_node->update(sub_output, 0, exp_output, 0);
                exp_nodes.push_back(exp_node);
                sum += exp_output;
            }
        }

        for (size_t i = 0; i < num_cols; ++i) {
            if (is_new_graph) {
                double div_output = exp_nodes[i]->output / sum;
                std::shared_ptr<Node> div_node = std::make_shared<Node>("divide", exp_nodes[i]->output, sum, div_output, 0);
                div_node->add_child(exp_nodes[i]);
                exp_nodes[i]->add_parent(div_node);

                node_list.push_back(div_node);
                row_result_ptr[i] = div_output;
            } else {
                auto div_node = node_list[row * num_cols + i];
                double div_output = exp_nodes[i]->output / sum;
                div_node->update(exp_nodes[i]->output, sum, div_output, 0);
                row_result_ptr[i] = div_node->output;

                // 노드 연결 확인 및 재설정
                auto exp_node = div_node->get_parents()[0];
                exp_node->update(div_node->input_value, 0.0, std::exp(div_node->input_value), 0);
            }
        }
    }

    return std::make_pair(result, node_list);
}
