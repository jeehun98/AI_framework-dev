#ifndef NODE_H
#define NODE_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include <unordered_set>
#include <iostream>

namespace py = pybind11;

class Node : public std::enable_shared_from_this<Node> {
public:
    // 기본 생성자
    Node() : operation("none"), input_value(0.0), weight_value(0.0), output(0.0), bias(0.0), grad_weight_total(0.0) {}

    // 생성자 오버로드
    Node(const std::string& op, double input, double weight, double out, double bias_value)
        : operation(op), input_value(input), weight_value(weight), output(out), bias(bias_value) {
        validate_operation();
    }

    Node(const std::string& op, double input, double out_val, double bias_value)
        : operation(op), input_value(input), output(out_val), bias(bias_value) {
        validate_operation();
    }

    // 부모 노드 추가
    void add_parent(std::shared_ptr<Node> parent) {
        if (std::find_if(parents.begin(), parents.end(),
            [&](const std::weak_ptr<Node>& wp) {
                return !wp.expired() && wp.lock() == parent;
            }) == parents.end()) {
            parents.emplace_back(parent);
        }
    }

    // 자식 노드 추가
    void add_child(std::shared_ptr<Node> child) {
        if (std::find(children.begin(), children.end(), child) == children.end()) {
            children.push_back(child);
            child->add_parent(shared_from_this());
        }
    }

    // 부모 노드 제거
    void remove_parent(std::shared_ptr<Node> parent) {
        parents.erase(std::remove_if(parents.begin(), parents.end(),
            [&](const std::weak_ptr<Node>& wp) {
                auto sp = wp.lock();
                return sp && sp == parent;
            }), parents.end());
    }

    // 자식 노드 제거
    void remove_child(std::shared_ptr<Node> child) {
        children.erase(std::remove(children.begin(), children.end(), child), children.end());
        child->remove_parent(shared_from_this());
    }

    // 리프 노드 찾기
    std::vector<std::shared_ptr<Node>> find_leaf_nodes() {
        std::vector<std::shared_ptr<Node>> leaf_nodes;
        find_leaf_nodes_recursive(shared_from_this(), leaf_nodes);
        return leaf_nodes;
    }

    // 리프 노드 재귀적으로 찾기
    void find_leaf_nodes_recursive(std::shared_ptr<Node> node, std::vector<std::shared_ptr<Node>>& leaf_nodes) {
        if (node->get_children().empty()) {
            leaf_nodes.push_back(node);
        } else {
            for (const auto& child : node->get_children()) {
                find_leaf_nodes_recursive(child, leaf_nodes);
            }
        }
    }

    // 노드 업데이트
    void update(double input, double weight, double out, double new_bias) {
        input_value = input;
        weight_value = weight;
        output = out;
        bias = new_bias;
    }

    // 자식 노드 반환
    std::vector<std::shared_ptr<Node>> get_children() const { return children; }

    // 부모 노드 반환
    std::vector<std::shared_ptr<Node>> get_parents() const {
        std::vector<std::shared_ptr<Node>> result;
        for (const auto& wp : parents) {
            if (auto sp = wp.lock()) {
                result.push_back(sp);
            }
        }
        return result;
    }

    // 그래디언트 계산
    std::pair<double, double> calculate_gradient(double upstream_gradient = 1.0) {
        auto it = operations().find(operation);
        if (it != operations().end()) {
            return it->second(input_value, weight_value, output, upstream_gradient);
        } else {
            throw std::runtime_error("Unsupported operation: " + operation + 
                                     ". Available operations: " + get_available_operations());
        }
    }

    // 역전파 메서드
    void backpropagate(double upstream_gradient = 1.0, std::unordered_set<Node*>* visited = nullptr) {
        if (!visited) {
            std::unordered_set<Node*> local_visited;
            backpropagate(upstream_gradient, &local_visited);
            return;
        }
        if (visited->find(this) != visited->end()) return;
        visited->insert(this);

        auto gradients = calculate_gradient(upstream_gradient);
        double grad_input = gradients.first;
        double grad_weight = gradients.second;
        grad_weight_total += grad_weight;

        for (auto& child : children) {
            child->backpropagate(grad_input, visited);
        }
    }

    // 가중치 업데이트
    void update_weights(double learning_rate, std::unordered_set<Node*>* visited = nullptr) {
        if (!visited) {
            std::unordered_set<Node*> local_visited;
            update_weights(learning_rate, &local_visited);
            return;
        }
        if (visited->find(this) != visited->end()) return;
        visited->insert(this);

        weight_value -= learning_rate * grad_weight_total;
        grad_weight_total = 0.0;

        for (auto& child : children) {
            child->update_weights(learning_rate, visited);
        }
    }

    // 그래프 출력
    void print_tree(const std::shared_ptr<Node>& node, int depth = 0, std::unordered_set<Node*>* visited = nullptr) {
        if (!node) return;
        if (!visited) {
            std::unordered_set<Node*> local_visited;
            print_tree(node, depth, &local_visited);
            return;
        }
        if (visited->find(node.get()) != visited->end()) {
            std::cout << std::string(depth * 4, ' ') << "Node: " << node->operation << " (already visited)" << std::endl;
            return;
        }
        visited->insert(node.get());

        std::cout << std::string(depth * 4, ' ') << "Node: " << node->operation 
                  << ", Weight: " << node->weight_value 
                  << ", Grad Total: " << node->grad_weight_total 
                  << ", Output: " << node->output << std::endl;

        for (const auto& child : node->children) {
            print_tree(child, depth + 1, visited);
        }
    }

private:
    void validate_operation() const {
        if (operations().find(operation) == operations().end()) {
            throw std::runtime_error("Invalid operation: " + operation);
        }
    }

    std::string get_available_operations() const {
        std::string available_operations;
        for (const auto& op : operations()) {
            available_operations += op.first + ", ";
        }
        if (!available_operations.empty()) {
            available_operations.pop_back();
            available_operations.pop_back();
        }
        return available_operations;
    }

    static const std::map<std::string, std::function<std::pair<double, double>(double, double, double, double)>>& operations() {
    static std::map<std::string, std::function<std::pair<double, double>(double, double, double, double)>> ops;
    if (ops.empty()) {
        ops["add"] = [](double input, double weight, double out, double upstream) {
            return std::make_pair(upstream, upstream);
        };
        ops["subtract"] = [](double input, double weight, double out, double upstream) {
            return std::make_pair(upstream, -upstream);
        };
        ops["multiply"] = [](double input, double weight, double out, double upstream) {
            return std::make_pair(upstream * weight, upstream * input);
        };
        ops["divide"] = [](double input, double weight, double out, double upstream) {
            if (weight == 0.0) {
                throw std::runtime_error("Division by zero.");
            }
            double grad_input = upstream / weight;
            double grad_weight = -upstream * input / (weight * weight);
            return std::make_pair(grad_input, grad_weight);
        };
        ops["exp"] = [](double input, double weight, double out, double upstream) {
            return std::make_pair(upstream * out, 0.0);
        };
        ops["square"] = [](double input, double weight, double out, double upstream) {
            return std::make_pair(2 * input * upstream, 0.0);
        };
        ops["reciprocal"] = [](double input, double weight, double out, double upstream) {
            if (input == 0.0) {
                throw std::runtime_error("Reciprocal of zero.");
            }
            return std::make_pair(-upstream / (input * input), 0.0);
        };
        ops["negate"] = [](double input, double weight, double out, double upstream) {
            return std::make_pair(-upstream, 0.0);
        };
        ops["max_pool"] = [](double input, double weight, double out, double upstream) {
            return std::make_pair((input == out) ? upstream : 0.0, 0.0);
        };
        ops["avg_pool"] = [](double input, double weight, double out, double upstream) {
            return std::make_pair(upstream / weight, 0.0);
        };
        ops["flatten"] = [](double input, double weight, double out, double upstream) {
            return std::make_pair(upstream, 0.0);
        };
        ops["bias"] = [](double input, double weight, double out, double upstream) {
            return std::make_pair(upstream, 0.0);
        };
    }
    return ops;
}


public:
    std::string operation;
    double input_value = 0.0;
    double weight_value = 0.0;
    double output = 0.0;
    double bias = 0.0;
    double grad_bias = 0.0;
    double grad_weight_total = 0.0;
    std::vector<std::weak_ptr<Node>> parents;
    std::vector<std::shared_ptr<Node>> children;
};

#endif // NODE_H
