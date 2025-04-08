#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <memory>
#include "../node/node.h"  // Node 클래스가 정의된 헤더 파일 포함

std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> relu(py::array_t<double> inputs, std::vector<std::shared_ptr<Node>> node_list = {});
std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> sigmoid(py::array_t<double> inputs, std::vector<std::shared_ptr<Node>> node_list = {});
std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> tanh_activation(py::array_t<double> inputs, std::vector<std::shared_ptr<Node>> node_list = {});
std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> leaky_relu(py::array_t<double> inputs, double alpha, std::vector<std::shared_ptr<Node>> node_list = {});
std::pair<py::array_t<double>, std::vector<std::shared_ptr<Node>>> softmax(py::array_t<double> inputs, std::vector<std::shared_ptr<Node>> node_list = {});

#endif // ACTIVATIONS_H
