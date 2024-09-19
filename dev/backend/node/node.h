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

namespace py = pybind11;

class Node {
public:
    std::string operation;              
    double input_a = 0.0;              
    double input_b = 0.0;              
    double output = 0.0;               
    double weight = 0.0;
    double grad_weight = 0.0;
    std::vector<std::shared_ptr<Node>> parents;   
    std::vector<std::shared_ptr<Node>> children;  

    Node(const std::string& op, double a, double b, double out, double weight)
        : operation(op), input_a(a), input_b(b), output(out), weight(weight) {
        init_operations(); 
        validate_operation();
    }

    Node(const std::string& op, double in_val, double out_val, double weight)
        : operation(op), input_a(in_val), output(out_val), weight(weight) {
        init_operations(); 
        validate_operation();
    }

    void add_parent(std::shared_ptr<Node> parent) {
        if (std::find(parents.begin(), parents.end(), parent) == parents.end()) {
            parents.push_back(parent);
        }
    }

    void add_child(std::shared_ptr<Node> child) {
        if (std::find(children.begin(), children.end(), child) == children.end()) {
            children.push_back(child);
        }
    }

    void remove_child(std::shared_ptr<Node> child) {
        children.erase(std::remove(children.begin(), children.end(), child), children.end());
    }

    void remove_parent(std::shared_ptr<Node> parent) {
        parents.erase(std::remove(parents.begin(), parents.end(), parent), parents.end());
    }

    void update(double a, double b, double out, double weight) {
        input_a = a;
        input_b = b;
        output = out;
        weight = weight;
    }

    std::vector<std::shared_ptr<Node>> get_children() const {
        return children;
    }

    std::vector<std::shared_ptr<Node>> get_parents() const {
        return parents;
    }

    std::pair<double, double> calculate_gradient(double upstream_gradient = 1.0) {
        auto it = operations.find(operation);
        if (it != operations.end()) {
            return it->second(input_a, input_b, output, upstream_gradient);
        } else {
            std::string available_operations;
            for (const auto& op : operations) {
                available_operations += op.first + ", ";
            }
            throw std::runtime_error("Unsupported operation: " + operation + ". Available operations: " + available_operations);
        }
    }

private:
    std::map<std::string, std::function<std::pair<double, double>(double, double, double, double)>> operations;

    void init_operations() {
        operations["add"] = [](double a, double b, double out, double upstream) {
            return std::make_pair(upstream, 0);
        };

        operations["subtract"] = [](double a, double b, double out, double upstream) {
            return std::make_pair(upstream, 0);
        };

        operations["multiply"] = [](double a, double b, double out, double upstream) {
            return std::make_pair(upstream * b, upstream * a);
        };

        operations["divide"] = [](double a, double b, double out, double upstream) {
            double grad_input = upstream / b;
            double grad_weight = -upstream * a / (b * b);
            return std::make_pair(grad_input, 0);
        };

        operations["exp"] = [](double a, double b, double out, double upstream) {
            return std::make_pair(upstream * out, 0.0);
        };

        operations["square"] = [](double a, double, double out, double upstream) {
            double grad_input = 2 * a * upstream;
            return std::make_pair(grad_input, 0.0);
        };

        operations["reciprocal"] = [](double a, double b, double out, double upstream) {
            double grad_input = -upstream / (a * a) * b;
            return std::make_pair(grad_input, 0.0);
        };

        operations["negate"] = [](double a, double, double out, double upstream) {
            double grad_input = -upstream;
            return std::make_pair(grad_input, 0);
        };
    }

    void validate_operation() const {
        if (operations.find(operation) == operations.end()) {
            throw std::runtime_error("Invalid operation: " + operation);
        }
    }
};

#endif // NODE_H
