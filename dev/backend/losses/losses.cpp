#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
#include "../node/node.h"

namespace py = pybind11;

std::pair<double, std::vector<std::shared_ptr<Node>>> mean_squared_error(
    py::array_t<double> y_true, 
    py::array_t<double> y_pred, 
    std::vector<std::shared_ptr<Node>> node_list = {}
) {
    py::buffer_info buf_true = y_true.request();
    py::buffer_info buf_pred = y_pred.request();

    if (buf_true.ndim != buf_pred.ndim) {
        throw std::invalid_argument("Input arrays must have the same dimensions");
    }

    if (buf_true.size != buf_pred.size) {
        throw std::invalid_argument("Input arrays must have the same size");
    }

    double* ptr_true = static_cast<double*>(buf_true.ptr);
    double* ptr_pred = static_cast<double*>(buf_pred.ptr);
    double mse = 0.0;

    bool is_new_graph = node_list.empty();

    for (ssize_t i = 0; i < buf_true.size; ++i) {
        double diff = ptr_true[i] - ptr_pred[i];

        if (is_new_graph) {
            // 새로운 그래프 생성
            std::shared_ptr<Node> diff_node = std::make_shared<Node>("subtract", ptr_true[i], ptr_pred[i], diff, 0);
            double squared_diff = diff * diff;
            std::shared_ptr<Node> square_node = std::make_shared<Node>("square", diff, squared_diff, 0);

            diff_node->add_parent(square_node);
            square_node->add_child(diff_node);

            node_list.push_back(square_node);
            mse += squared_diff;
        } else {
            // 기존 그래프 업데이트
            auto square_node = node_list[i];
            auto diff_node = square_node->get_children()[0];

            diff_node->update(ptr_true[i], ptr_pred[i], diff, 0);
            double squared_diff = diff * diff;
            square_node->update(diff, 0.0, squared_diff, 0);
            mse += squared_diff;
        }
    }
    mse = mse / buf_true.size;

    return std::make_pair(mse, node_list);
}

std::pair<double, std::vector<std::shared_ptr<Node>>> cross_entropy_loss(
    py::array_t<double> y_true, 
    py::array_t<double> y_pred, 
    std::vector<std::shared_ptr<Node>> node_list = {}
) {
    py::buffer_info buf_true = y_true.request(), buf_pred = y_pred.request();

    if (buf_true.ndim != buf_pred.ndim) {
        throw std::invalid_argument("Input arrays must have the same dimensions");
    }

    if (buf_true.size != buf_pred.size) {
        throw std::invalid_argument("Input arrays must have the same size");
    }

    double* ptr_true = static_cast<double*>(buf_true.ptr);
    double* ptr_pred = static_cast<double*>(buf_pred.ptr);
    double loss = 0.0;

    bool is_new_graph = node_list.empty();

    for (size_t i = 0; i < buf_true.size; ++i) {
        if (is_new_graph) {
            // 새로운 그래프 생성
            double log_pred = std::log(ptr_pred[i]);
            std::shared_ptr<Node> log_pred_node = std::make_shared<Node>("log", ptr_pred[i], log_pred, 0);

            double term1 = ptr_true[i] * log_pred;
            std::shared_ptr<Node> mul_node1 = std::make_shared<Node>("multiply", ptr_true[i], log_pred, term1, 0);

            log_pred_node->add_parent(mul_node1);
            mul_node1->add_child(log_pred_node);

            double constant_value = 1.0;
            double one_minus_pred = constant_value - ptr_pred[i];
            std::shared_ptr<Node> one_minus_pred_node = std::make_shared<Node>("subtract", constant_value, ptr_pred[i], one_minus_pred, 0);

            double log_one_minus_pred = std::log(one_minus_pred);
            std::shared_ptr<Node> log_one_minus_pred_node = std::make_shared<Node>("log", one_minus_pred, log_one_minus_pred, 0);

            log_one_minus_pred_node->add_child(one_minus_pred_node);
            one_minus_pred_node->add_parent(log_one_minus_pred_node);

            double one_minus_true = constant_value - ptr_true[i];
            std::shared_ptr<Node> one_minus_true_node = std::make_shared<Node>("subtract", constant_value, ptr_true[i], one_minus_true, 0);

            double term2 = one_minus_true * log_one_minus_pred;
            std::shared_ptr<Node> mul_node2 = std::make_shared<Node>("multiply", one_minus_true, log_one_minus_pred, term2, 0);

            mul_node2->add_child(one_minus_true_node);
            mul_node2->add_child(log_one_minus_pred_node);

            log_one_minus_pred_node->add_parent(mul_node2);
            one_minus_true_node->add_parent(mul_node2);

            double add_term = term1 + term2;
            std::shared_ptr<Node> add_term_node = std::make_shared<Node>("add", term1, term2, add_term, 0);

            add_term_node->add_child(mul_node1);
            add_term_node->add_child(mul_node2);

            mul_node1->add_parent(add_term_node);
            mul_node2->add_parent(add_term_node);

            double sample_loss = -add_term;
            std::shared_ptr<Node> neg_node = std::make_shared<Node>("negate", add_term, sample_loss, 0);

            neg_node->add_child(add_term_node);
            add_term_node->add_parent(neg_node);

            node_list.push_back(neg_node);
            loss += sample_loss;
        } else {
            // 기존 그래프 업데이트
            auto neg_node = node_list[i];
            auto add_term_node = neg_node->get_children()[0];
            auto mul_node1 = add_term_node->get_children()[0];
            auto log_pred_node = mul_node1->get_children()[0];

            double log_pred = std::log(ptr_pred[i]);
            log_pred_node->update(ptr_pred[i], 0.0, log_pred, 0);
            double term1 = ptr_true[i] * log_pred;
            mul_node1->update(ptr_true[i], log_pred, term1, 0);

            auto mul_node2 = add_term_node->get_children()[1];
            auto one_minus_true_node = mul_node2->get_children()[0];
            auto log_one_minus_pred_node = mul_node2->get_children()[1];
            auto one_minus_pred_node = log_one_minus_pred_node->get_children()[0];

            double one_minus_pred = 1.0 - ptr_pred[i];
            one_minus_pred_node->update(1.0, ptr_pred[i], one_minus_pred, 0);
            double log_one_minus_pred = std::log(one_minus_pred);
            log_one_minus_pred_node->update(one_minus_pred, 0.0, log_one_minus_pred, 0);

            double one_minus_true = 1.0 - ptr_true[i];
            one_minus_true_node->update(1.0, ptr_true[i], one_minus_true, 0);
            double term2 = one_minus_true * log_one_minus_pred;
            mul_node2->update(one_minus_true, log_one_minus_pred, term2, 0);

            double add_term = term1 + term2;
            add_term_node->update(term1, term2, add_term, 0);

            double sample_loss = -add_term;
            neg_node->update(add_term, 0.0, sample_loss, 0);

            loss += sample_loss;
        }
    }

    return std::make_pair(loss / buf_true.size, node_list);
}
