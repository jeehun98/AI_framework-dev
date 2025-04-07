# dev/cal_graph/activations.py

from .node import Node

def build_sigmoid_node(input_node):
    # Sigmoid(x) = 1 / (1 + exp(-x))
    neg_node = Node("neg")
    neg_node.add_parent(input_node)

    exp_node = Node("exp")
    exp_node.add_parent(neg_node)

    one_node = Node("const", input_value=1.0)

    add_node = Node("add")
    add_node.add_parent(one_node)
    add_node.add_parent(exp_node)

    reciprocal_node = Node("reciprocal")
    reciprocal_node.add_parent(add_node)

    return reciprocal_node


def build_tanh_node(input_node):
    # tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    exp_pos = Node("exp")
    exp_pos.add_parent(input_node)

    neg_node = Node("neg")
    neg_node.add_parent(input_node)

    exp_neg = Node("exp")
    exp_neg.add_parent(neg_node)

    numerator = Node("subtract")
    numerator.add_parent(exp_pos)
    numerator.add_parent(exp_neg)

    denominator = Node("add")
    denominator.add_parent(exp_pos)
    denominator.add_parent(exp_neg)

    divide_node = Node("divide")
    divide_node.add_parent(numerator)
    divide_node.add_parent(denominator)

    return divide_node


def build_relu_node(input_node):
    # ReLU(x) = x * sigmoid(10 * x) 로 근사
    scale_node = Node("const", input_value=10.0)

    scale_mul_node = Node("multiply")
    scale_mul_node.add_parent(input_node)
    scale_mul_node.add_parent(scale_node)

    sigmoid_node = build_sigmoid_node(scale_mul_node)

    relu_node = Node("multiply")
    relu_node.add_parent(input_node)
    relu_node.add_parent(sigmoid_node)

    return relu_node
