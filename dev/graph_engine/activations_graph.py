from .node import Node

def build_sigmoid_node():
    """
    Sigmoid(x) = 1 / (1 + exp(-x))
    구조:
        reciprocal
          └── add
              ├── const(1.0)
              └── exp
                   └── neg
    """
    neg_node = Node("neg")

    exp_node = Node("exp")
    exp_node.add_child(neg_node)

    one_node = Node("const", input_value=1.0)

    add_node = Node("add")
    add_node.add_child(one_node)
    add_node.add_child(exp_node)

    reciprocal_node = Node("reciprocal")
    reciprocal_node.add_child(add_node)

    return reciprocal_node, [neg_node]


def build_tanh_node():
    """
    tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    구조:
        divide
          ├── subtract
          │     ├── exp(x)
          │     └── exp(-x)
          └── add
                ├── exp(x)
                └── exp(-x)
    """
    exp_pos = Node("exp")

    neg_node = Node("neg")
    exp_neg = Node("exp")
    exp_neg.add_child(neg_node)

    numerator = Node("subtract")
    numerator.add_child(exp_pos)
    numerator.add_child(exp_neg)

    denominator = Node("add")
    denominator.add_child(exp_pos)
    denominator.add_child(exp_neg)

    divide_node = Node("divide")
    divide_node.add_child(numerator)
    divide_node.add_child(denominator)

    return divide_node, [exp_pos, neg_node]


def build_relu_node():
    """
    ReLU(x) ≈ x * sigmoid(10 * x)
    구조:
        multiply (ReLU)
          ├── sigmoid
          │     └── reciprocal
          │         └── add
          │             ├── const(1.0)
          │             └── exp
          │                 └── neg
          │                     └── multiply
          │                         ├── const(10.0)
          │                         └── x
          └── x
    """
    x_node = Node("const", input_value=0.0)

    scale_node = Node("const", input_value=10.0)

    scale_mul_node = Node("multiply")
    scale_mul_node.add_child(x_node)
    scale_mul_node.weight_value = scale_node.input_value

    neg_node = Node("neg")
    neg_node.add_child(scale_mul_node)

    exp_node = Node("exp")
    exp_node.add_child(neg_node)

    one_node = Node("const", input_value=1.0)

    add_node = Node("add")
    add_node.add_child(one_node)
    add_node.add_child(exp_node)

    reciprocal_node = Node("reciprocal")
    reciprocal_node.add_child(add_node)

    relu_node = Node("multiply")
    relu_node.add_child(reciprocal_node)
    relu_node.add_child(x_node)

    return relu_node, [x_node]
