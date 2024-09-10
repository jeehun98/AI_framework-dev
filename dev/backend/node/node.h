#ifndef NODE_H
#define NODE_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <functional>
#include <stdexcept>

namespace py = pybind11;

class Node {
public:
    std::string operation;             // 연산의 이름
    double input_a = 0.0;              // 첫 번째 입력 값 (선택적)
    double input_b = 0.0;              // 두 번째 입력 값 (선택적)
    double output = 0.0;               // 출력 값
    std::vector<std::shared_ptr<Node>> parents;   // 부모 노드들
    std::vector<std::shared_ptr<Node>> children;  // 자식 노드들

    // 두 개의 입력을 받는 노드를 위한 생성자
    Node(const std::string& op, double a, double b, double out)
        : operation(op), input_a(a), input_b(b), output(out) {}

    // 단일 입력을 받는 노드를 위한 생성자
    Node(const std::string& op, double in_val, double out_val)
        : operation(op), input_a(in_val), output(out_val) {}

    // 부모 노드를 추가하는 함수
    void add_parent(std::shared_ptr<Node> parent) {
        parents.push_back(parent);
    }

    // 자식 노드를 추가하는 함수
    void add_child(std::shared_ptr<Node> child) {
        children.push_back(child);
    }

    std::pair<double, double> calculate_gradient(double upstrema_gradient = 1.0){
        auto it = operations.find(operation);
        if(it != operations.end()){
            return it->second(input_a, input_b, output, upstrema_gradient);
        }
    }

private:
    std::map<std::string, std::function<std::pair<double, double>(double, double, double, double)>> operations;

    void init_operations(){
        operations["add"] = [](double a, double b, double out, double upstream){
            return std::make_pair(upstream, upstream);
        };

        operations["substract"] = [](double a, double b, double out, double upstream){
            return std::make_pair(upstream, -upstream);
        };

        operations["multiply"] = [](double a, double b, double out, double upstream){
            return std::make_pair(upstream * b, upstream * a);
        };

        operations["divide"] = [](double a, double b, double out, double upstream){
            double grad_a = upstream / b;
            double grad_b = -upstream * a / (b * b);
            return std::make_pair(grad_a, grad_b);
        };

        operations["exp"] = [](double a, double b, double out, double upstream){
            return std::make_pair(upstream * out, 0.0);
        };
    }
};

#endif // NODE_H
