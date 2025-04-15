# dev/cal_graph/activations_graph.py
# 활성화 함수 별 계산 그래프 구조조

from .node import Node

def build_sigmoid_node():
    # Sigmoid(x) = 1 / (1 + exp(-x))
    neg_node = Node("neg")  # 부모 없이 구성

    exp_node = Node("exp")
    exp_node.add_parent(neg_node)

    one_node = Node("const", input_value=1.0)

    add_node = Node("add")
    add_node.add_parent(one_node)
    add_node.add_parent(exp_node)

    reciprocal_node = Node("reciprocal")
    reciprocal_node.add_parent(add_node)

    return reciprocal_node  # 루트 노드만 반환

def build_tanh_node():
    exp_pos = Node("exp")  # 입력 없음

    neg_node = Node("neg")

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

def build_relu_node():
    scale_node = Node("const", input_value=10.0)

    scale_mul_node = Node("multiply")
    scale_mul_node.add_parent(scale_node)

    sigmoid_node = build_sigmoid_node()

    relu_node = Node("multiply")
    relu_node.add_parent(sigmoid_node)

    return relu_node
