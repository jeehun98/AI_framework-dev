#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <map>
#include <unordered_set>
#include "../node/node.h"  // Node 클래스 헤더 파일 경로 포함

namespace py = pybind11;

class SGD {
public:
    SGD(double learning_rate) : learning_rate(learning_rate) {}

    void update(Node& node, double lr) {  // lr 인자를 추가로 받아 동적으로 적용
        double updated_weight = node.get_weight() - lr * node.get_gradient();
        node.set_weight(updated_weight);
    }

    // 루트 노드에서 시작하여 모든 하위 노드의 가중치를 갱신
    void update_all_weights(std::shared_ptr<Node> root, double lr) {  // lr 인자를 추가로 받음
        std::unordered_set<Node*> visited; // 방문한 노드를 추적하기 위한 집합
        update_all_weights_recursive(root, visited, lr);  // 재귀 메서드에 lr 전달
    }

private:
    double learning_rate;

    // 재귀적으로 모든 노드를 방문하여 가중치를 업데이트
    void update_all_weights_recursive(std::shared_ptr<Node> node, std::unordered_set<Node*>& visited, double lr) {
        if (!node || visited.find(node.get()) != visited.end()) return; // 이미 방문한 노드라면 반환
        visited.insert(node.get());

        // 현재 노드의 가중치를 업데이트
        update(*node, lr);  // 동적으로 전달된 lr 사용

        // 자식 노드들에 대해 재귀적으로 가중치 업데이트
        for (auto& child : node->get_children()) {
            update_all_weights_recursive(child, visited, lr);  // 자식 노드에 lr 전달
        }
    }
};


class Adam {
public:
    Adam(double learning_rate, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        : learning_rate(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}

    void update(Node& node) {
        size_t node_id = reinterpret_cast<size_t>(&node);
        double gradient = node.get_gradient();

        // 모멘트 벡터 업데이트
        m[node_id] = beta1 * m[node_id] + (1.0 - beta1) * gradient;
        v[node_id] = beta2 * v[node_id] + (1.0 - beta2) * gradient * gradient;

        // 모멘트 벡터 보정
        double m_hat = m[node_id] / (1.0 - std::pow(beta1, t + 1));
        double v_hat = v[node_id] / (1.0 - std::pow(beta2, t + 1));

        // 가중치 업데이트
        double updated_weight = node.get_weight() - learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        node.set_weight(updated_weight);
    }

    // 루트 노드에서 시작하여 모든 하위 노드의 가중치를 갱신
    void update_all_weights(std::shared_ptr<Node> root) {
        std::unordered_set<Node*> visited; // 방문한 노드를 추적하기 위한 집합
        t++; // 타임스텝 증가
        update_all_weights_recursive(root, visited);
    }

private:
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    int t; // 타임스텝
    std::map<size_t, double> m; // 1차 모멘트 벡터
    std::map<size_t, double> v; // 2차 모멘트 벡터

    // 재귀적으로 모든 노드를 방문하여 가중치를 업데이트
    void update_all_weights_recursive(std::shared_ptr<Node> node, std::unordered_set<Node*>& visited) {
        if (!node || visited.find(node.get()) != visited.end()) return; // 이미 방문한 노드라면 반환
        visited.insert(node.get());

        // 현재 노드의 가중치를 업데이트
        update(*node);

        // 자식 노드들에 대해 재귀적으로 가중치 업데이트
        for (auto& child : node->get_children()) {
            update_all_weights_recursive(child, visited);
        }
    }
};
