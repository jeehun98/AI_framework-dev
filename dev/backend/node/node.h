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
#include <stack>
#include <unordered_set>

namespace py = pybind11;

class Node : public std::enable_shared_from_this<Node> {
public:
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
        // 중복 추가 방지
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

    // 자식 노드 제거
    void remove_child(std::shared_ptr<Node> child) {
        children.erase(std::remove(children.begin(), children.end(), child), children.end());
        child->remove_parent(shared_from_this());
    }

    // 부모 노드 제거
    void remove_parent(std::shared_ptr<Node> parent) {
        parents.erase(std::remove_if(parents.begin(), parents.end(),
            [&](const std::weak_ptr<Node>& wp) {
                auto sp = wp.lock();
                return sp && sp == parent;
            }), parents.end());
    }

    // 노드 업데이트
    void update(double input, double weight, double out, double new_bias) {
        input_value = input;
        weight_value = weight;
        output = out;
        bias = new_bias;
    }

    // 자식 노드 반환
    std::vector<std::shared_ptr<Node>> get_children() const {
        return children;
    }

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

    // 역전파 메서드 (루트 노드에서 자식 노드로 내려가는 방식)
    // 순환 구조 방지를 위해 visited 집합을 추가
    void backpropagate(double upstream_gradient = 1.0, std::unordered_set<Node*>* visited = nullptr) {
        // 방문 집합 초기화
        if (!visited) {
            std::unordered_set<Node*> local_visited;
            backpropagate(upstream_gradient, &local_visited);
            return;
        }

        // 현재 노드가 이미 방문된 노드라면 재귀 호출을 중단
        if (visited->find(this) != visited->end()) {
            
            return;
        }

        // 현재 노드를 방문한 것으로 기록
        visited->insert(this);

        // 1. 현재 노드에서 그래디언트 계산
        auto gradients = calculate_gradient(upstream_gradient);
        double grad_input = gradients.first;   // 입력값에 대한 그래디언트
        double grad_weight = gradients.second; // 가중치에 대한 그래디언트

        // 2. 가중치에 대한 그래디언트 누적
        grad_weight_total += grad_weight;

        // 3. 자식 노드로 그래디언트 전달
        for (auto& child : children) {
            // 자식 노드로 현재 노드의 grad_input 값을 전달
            child->backpropagate(grad_input, visited);
        }
    }


    void update_weights(double learning_rate, std::unordered_set<Node*>* visited = nullptr) {
        if (!visited) {
            std::unordered_set<Node*> local_visited;
            update_weights(learning_rate, &local_visited);
            return;
        }

        if (visited->find(this) != visited->end()) {
            return;
        }
        visited->insert(this);

        // 현재 노드의 가중치 업데이트
        weight_value -= learning_rate * grad_weight_total;
        grad_weight_total = 0.0;

        // 자식 노드에 대해 재귀적으로 가중치 업데이트
        for (auto& child : children) {
            child->update_weights(learning_rate, visited);
        }
    }

public:
    std::string operation;
    double input_value = 0.0;          // 입력값
    double weight_value = 0.0;         // 가중치 값
    double output = 0.0;
    double bias = 0.0;                 // 바이어스
    double grad_bias = 0.0;            // 바이어스에 대한 그래디언트
    double grad_weight_total = 0.0;    // 가중치에 대한 누적 그래디언트
    std::vector<std::weak_ptr<Node>> parents;
    std::vector<std::shared_ptr<Node>> children;

    // 연산자 맵을 정적 멤버로 선언하여 모든 인스턴스에서 공유
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
                double grad_input = 2 * input * upstream;
                return std::make_pair(grad_input, 0.0);
            };

            ops["reciprocal"] = [](double input, double weight, double out, double upstream) {
                if (input == 0.0) {
                    throw std::runtime_error("Reciprocal of zero.");
                }
                double grad_input = -upstream / (input * input);
                return std::make_pair(grad_input, 0.0);
            };

            ops["negate"] = [](double input, double weight, double out, double upstream) {
                double grad_input = -upstream;
                return std::make_pair(grad_input, 0.0);
            };
        }
        return ops;
    }

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
};

#endif // NODE_H
