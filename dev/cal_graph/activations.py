from dev.cal_graph.node import Node

def build_sigmoid_node(input_node):
    neg_node = Node("neg")
    neg_node.add_parent(input_node)

    exp_node = Node("exp")
    exp_node.add_parent(neg_node)

    add_one_node = Node("add", input_value=1.0)
    add_one_node.add_parent(exp_node)

    reciprocal_node = Node("reciprocal")
    reciprocal_node.add_parent(add_one_node)

    return reciprocal_node


def build_tanh_node(input_node):
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

    tanh_node = Node("divide")
    tanh_node.add_parent(numerator)
    tanh_node.weight_value = 1.0  # 나누는 weight로 사용

    return tanh_node


def build_relu_node(input_node):
    """
    ReLU(x) = max(0, x)
    간단한 방식으로 근사: ReLU(x) = x * sigmoid(large * x)
    """
    scale = 10.0  # large 값 (ReLU 근사)

    scale_node = Node("multiply", input_node.input_value, scale)
    scale_node.add_parent(input_node)

    sigmoid_node = build_sigmoid_node(scale_node)

    relu_node = Node("multiply")
    relu_node.add_parent(input_node)
    relu_node.add_parent(sigmoid_node)

    return relu_node
