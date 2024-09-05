#ifndef NODE_H
#define NODE_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <memory>
#include <string>

namespace py = pybind11;

class Node {
public:
    std::string operation;
    double input_a;
    double input_b;
    double output;
    std::vector<std::shared_ptr<Node>> parents;  // 부모 노드들
    std::vector<std::shared_ptr<Node>> children; // 자식 노드들

    Node(const std::string& op, double a, double b, double out)
        : operation(op), input_a(a), input_b(b), output(out) {}

    // 부모 노드를 추가하는 함수
    void add_parent(std::shared_ptr<Node> parent) {
        parents.push_back(parent);
    }

    // 자식 노드를 추가하는 함수
    void add_child(std::shared_ptr<Node> child) {
        children.push_back(child);
    }
};

#endif // NODE_H
