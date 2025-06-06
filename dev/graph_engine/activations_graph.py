from .node import Node

# ✅ Sigmoid 계산 그래프
def build_sigmoid_node(result):
    """
    Sigmoid(x) = 1 / (1 + exp(-x))

    🔹 계산 그래프 구조
        [reciprocal]
             │
           [add]
          /     \
    [const(1.0)] [exp]
                      │
                   [neg]
                      │
                      x
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

    reciprocal_node.output = result

    return reciprocal_node, [neg_node]

# ------------------------------------------------

# ✅ Tanh 계산 그래프
def build_tanh_node(result):
    """
    tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    🔹 계산 그래프 구조
            [divide]
             /     \
       [subtract] [add]
        /     \    /   \
    [exp(x)] [exp(-x)] [exp(x)] [exp(-x)]
                    │
                 [neg]
                    │
                    x
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

    divide_node.output = result

    return divide_node, [exp_pos, neg_node]

# ------------------------------------------------

# ✅ ReLU 계산 그래프
def build_relu_node(result):
    """
    ReLU(x) ≈ x * sigmoid(10 * x)

    🔹 계산 그래프 구조
           [multiply]   ← ReLU 근사
             /     \
     [reciprocal]     x
          │
        [add]
       /     \
 [const(1.0)] [exp]
                   │
                [neg]
                   │
              [multiply]
               /      \
        [const(10.0)]  x
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

    relu_node.output = result

    return relu_node, [x_node]
