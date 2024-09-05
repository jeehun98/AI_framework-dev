#include "node.h"

Node::Node(const std::string& op, const std::vector<py::array_t<double>>& in, py::array_t<double> out)
    : operation(op), inputs(in), output(out) {}
