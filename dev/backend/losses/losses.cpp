#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
#include "../node/node.h"

namespace py = pybind11;

double mean_squared_error(py::array_t<double> y_true, py::array_t<double> y_pred) {
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

    // 노드 리스트 생성
    std::vector<std::shared_ptr<Node>> node_list;

    for (ssize_t i = 0; i < buf_true.size; ++i) {
        double diff = ptr_true[i] - ptr_pred[i];

        std::shared_ptr<Node> diff_node = std::make_shared<Node>("substract", ptr_true[i], ptr_pred[i], diff);

        double squared_diff = diff * diff;

        std::shared_ptr<Node> square_node = std::make_shared<Node>("square", diff, squared_diff);

        diff_node->add_parent(square_node);
        square_node->add_child(diff_node);

        node_list.push_back(square_node);

        mse += squared_diff;
    }

    return mse / buf_true.size;
}

double cross_entropy_loss(py::array_t<double> y_true, py::array_t<double> y_pred) {
    py::buffer_info buf_true = y_true.request(), buf_pred = y_pred.request();

    if (buf_true.ndim != buf_pred.ndim) {
        throw std::invalid_argument("Input arrays must have the same dimensions");
    }

    if (buf_true.size != buf_pred.size) {
        throw std::invalid_argument("Input arrays must have the same size");
    }

    // 노드 리스트 생성
    std::vector<std::shared_ptr<Node>> node_list;

    double* ptr_true = static_cast<double*>(buf_true.ptr);
    double* ptr_pred = static_cast<double*>(buf_pred.ptr);
    double loss = 0.0;

    for (size_t i = 0; i < buf_true.size; ++i) {
        double log_pred = std::log(ptr_pred[i]);
        std::shared_ptr<Node> log_pred_node = std::make_shared<Node>("log", ptr_pred[i], log_pred);
        
        double term1 = ptr_true[i] * log_pred;
        std::shared_ptr<Node> mul_node1 = std::make_shared<Node>("multiply", ptr_true[i], log_pred, term1);
        
        log_pred_node->add_parent(mul_node1);
        mul_node1->add_child(log_pred_node);

        double constant_value = 1.0;
        double one_minus_pred = constant_value - ptr_pred[i];
        std::shared_ptr<Node> one_minus_pred_node =  std::make_shared<Node>("substract", constant_value, ptr_pred[i], one_minus_pred);

        double log_one_minus_pred = std::log(one_minus_pred);
        std::shared_ptr<Node> log_one_minus_pred_node = std::make_shared<Node>("log", one_minus_pred, log_one_minus_pred);

        log_one_minus_pred_node->add_child(one_minus_pred_node);
        one_minus_pred_node->add_parent(log_one_minus_pred_node);


        double one_minus_true = constant_value - ptr_true[i];
        std::shared_ptr<Node> one_minus_true_node = std::make_shared<Node>("substract", constant_value, ptr_true[i], one_minus_true);
        
        double term2 = one_minus_true * log_one_minus_pred;
        std::shared_ptr<Node> mul_node2 = std::make_shared<Node>("multiply", one_minus_true, log_one_minus_pred, term2);
        
        mul_node2->add_child(one_minus_true_node);
        mul_node2->add_child(log_one_minus_pred_node);

        log_one_minus_pred_node->add_parent(mul_node2);
        one_minus_true_node->add_parent(mul_node2);

        double add_term = term1 + term2;
        std::shared_ptr<Node> add_term_node = std::make_shared<Node>("add", term1, term2, add_term);

        add_term_node->add_child(mul_node1);
        add_term_node->add_child(mul_node2);

        mul_node1->add_parent(add_term_node);
        mul_node2->add_parent(add_term_node);

        double sample_loss = -add_term;
        std::shared_ptr<Node> neg_node = std::make_shared<Node>("negate", add_term, sample_loss);

        neg_node->add_child(add_term_node);
        add_term_node->add_parent(neg_node);

        loss += sample_loss;
    }

    return loss / buf_true.size;
}
