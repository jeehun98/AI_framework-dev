from .node import Node

# ✅ MSE 계산 그래프
def build_mse_node():
    """
    MSE = mean((y_true - y_pred)^2)
    구조:
          mean
           │
         square
           │
        subtract
        /      \
     const    const
    """
    y_true_node = Node("const")
    y_pred_node = Node("const")

    sub_node = Node("subtract")
    sub_node.add_child(y_true_node)
    sub_node.add_child(y_pred_node)

    square_node = Node("square")
    square_node.add_child(sub_node)

    mean_node = Node("mean")
    mean_node.add_child(square_node)

    return mean_node, [y_pred_node]

# ------------------------------------------------

# ✅ Binary Crossentropy 계산 그래프
def build_binary_crossentropy_node():
    """
    BCE = - [ y * log(p) + (1 - y) * log(1 - p) ]
    구조:
           neg
            │
          add
         /    \
      mul      mul
     /   \    /   \
  const log  sub  log
         │   / \    │
       const 1 const sub
                      / \
                   const const
    """
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

    return neg_node, [y_pred]

# ------------------------------------------------

# ✅ Categorical Crossentropy 계산 그래프
def build_categorical_crossentropy_node(num_classes=3):
    """
    CCE = - sum( y_i * log(p_i) )
    구조:
         neg
          │
         add
        /   \
     ...   mul
           /  \
        const  log
                 │
               const
    (num_classes 만큼 반복)
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

    sum_node = mul_nodes[0]
    for node in mul_nodes[1:]:
        add_node = Node("add")
        add_node.add_child(sum_node)
        add_node.add_child(node)
        sum_node = add_node

    neg_node = Node("neg")
    neg_node.add_child(sum_node)

    # 리프 노드는 모든 y_true, y_pred
    leaf_nodes = []
    for mul in mul_nodes:
        # ✅ y_pred만 추출
        leaf_nodes.append(mul.children[1])  # mul.children = [y_true, log_p]

    return neg_node, leaf_nodes
