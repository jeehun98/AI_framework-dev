from .node import Node

# ✅ MSE 계산 그래프
def build_mse_node(num_outputs, result=None):
    """
    MSE = mean( sum( (y_true_i - y_pred_i)^2 ) )

    🔹 계산 그래프 구조 (num_outputs = 3 예시)
            [mean]
              │
            [sum]
           /  |  \
      [square][square][square]
           │      │       │
       [subtract][subtract][subtract]
         /   \     /   \     /   \
     [const][const] ...
     (y_true  y_pred)
    """
    square_nodes = []

    for _ in range(num_outputs):
        y_true_node = Node("const")
        y_pred_node = Node("const")

        sub_node = Node("subtract")
        sub_node.add_child(y_true_node)
        sub_node.add_child(y_pred_node)

        square_node = Node("square")
        square_node.add_child(sub_node)

        square_nodes.append(square_node)

    sum_node = Node("sum")
    for node in square_nodes:
        sum_node.add_child(node)

    mean_node = Node("mean")
    mean_node.add_child(sum_node)

    # ✅ CUDA 결과값 저장
    if result is not None:
        mean_node.output = result

    leaf_nodes = [sq.children[0].children[1] for sq in square_nodes]  # 각 y_pred 노드

    return mean_node, leaf_nodes

# ------------------------------------------------

# ✅ Binary Crossentropy 계산 그래프
def build_binary_crossentropy_node(num_outputs=1, result=None):
    """
    BCE = mean( - [ y * log(p) + (1 - y) * log(1 - p) ] )

    🔹 계산 그래프 구조 (num_outputs = 2 예시)
              [mean]
                │
              [sum]
              /   \
           [neg]  [neg]
             │      │
           [add]   [add]
          /     \  /     \
      [mul]   [mul]   [mul]
      /  \     /  \     ...
   [const][log] ...
           │
         [const]
    """
    nodes = []

    for _ in range(num_outputs):
        y_true = Node("const")
        y_pred = Node("const")
        one_const = Node("const", input_value=1.0)

        log_p = Node("log")
        log_p.add_child(y_pred)

        mul1 = Node("multiply")
        mul1.add_child(y_true)
        mul1.add_child(log_p)

        one_minus_y = Node("subtract")
        one_minus_y.add_child(one_const)
        one_minus_y.add_child(y_true)

        one_minus_p = Node("subtract")
        one_minus_p.add_child(one_const)
        one_minus_p.add_child(y_pred)

        log_one_minus_p = Node("log")
        log_one_minus_p.add_child(one_minus_p)

        mul2 = Node("multiply")
        mul2.add_child(one_minus_y)
        mul2.add_child(log_one_minus_p)

        add_node = Node("add")
        add_node.add_child(mul1)
        add_node.add_child(mul2)

        neg_node = Node("neg")
        neg_node.add_child(add_node)

        nodes.append(neg_node)

    if num_outputs == 1:
        final_node = nodes[0]
    else:
        sum_node = Node("sum")
        for n in nodes:
            sum_node.add_child(n)

        final_node = Node("mean")
        final_node.add_child(sum_node)

    # ✅ CUDA 결과값 저장
    if result is not None:
        final_node.output = result

    leaf_nodes = []
    for n in nodes:
        leaf_nodes.append(n.children[0].children[1])  # neg → add → mul1 → y_pred

    return final_node, leaf_nodes

# ------------------------------------------------

# ✅ Categorical Crossentropy 계산 그래프
def build_categorical_crossentropy_node(num_classes=3, result=None):
    """
    CCE = - sum( y_i * log(p_i) )

    🔹 계산 그래프 구조 (num_classes = 3 예시)
              [neg]
                │
              [sum]
             /   |   \
         [mul] [mul] [mul]
         /  \   /  \   /  \
     [const][log] ...
             │
          [const]
    """
    mul_nodes = []

    for _ in range(num_classes):
        y_true = Node("const")
        y_pred = Node("const")

        log_p = Node("log")
        log_p.add_child(y_pred)

        mul = Node("multiply")
        mul.add_child(y_true)
        mul.add_child(log_p)

        mul_nodes.append(mul)

    sum_node = Node("sum")
    for node in mul_nodes:
        sum_node.add_child(node)

    neg_node = Node("neg")
    neg_node.add_child(sum_node)

    # ✅ CUDA 결과값 저장
    if result is not None:
        neg_node.output = result

    leaf_nodes = [mul.children[1].children[0] for mul in mul_nodes]  # log → y_pred

    return neg_node, leaf_nodes

# ------------------------------------------------

# ✅ 손실 함수 매핑 딕셔너리
loss_graph_builders = {
    "mse": build_mse_node,
    "binary_crossentropy": build_binary_crossentropy_node,
    "categorical_crossentropy": build_categorical_crossentropy_node,
}
