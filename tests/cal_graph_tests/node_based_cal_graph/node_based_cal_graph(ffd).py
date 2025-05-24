import numpy as np
import pandas as pd

# 🔢 실험 설정: (1,5) 입력 → Dense(5→3) + Sigmoid → Dense(3→4)
x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])  # shape (1, 5)
W1 = np.ones((5, 3)) * 0.1
b1 = np.zeros((1, 3))
W2 = np.ones((3, 4)) * 0.2
b2 = np.zeros((1, 4))

# 순전파
z1 = x @ W1 + b1                     # shape (1, 3)
a1 = 1 / (1 + np.exp(-z1))           # sigmoid activation
z2 = a1 @ W2 + b2                    # shape (1, 4)
output = z2

# 역전파 초기 gradient
grad_output = np.ones_like(output)  # shape (1, 4)

# 역전파: Dense2 → Sigmoid → Dense1
grad_W2 = a1.T @ grad_output               # shape (3,4)
grad_a1 = grad_output @ W2.T              # shape (1,3)
grad_z1 = grad_a1 * a1 * (1 - a1)         # shape (1,3)
grad_W1 = x.T @ grad_z1                   # shape (5,3)

# 결과 정리
result_data = {
    "output (1x4)": output.flatten(),
    "grad_output (1x4)": grad_output.flatten(),
    "z1 (1x3)": z1.flatten(),
    "a1 = sigmoid(z1) (1x3)": a1.flatten(),
    "grad_z1": grad_z1.flatten(),
}

grad_W1_df = pd.DataFrame(grad_W1, columns=[f"W1_grad_out{j}" for j in range(3)])
grad_W2_df = pd.DataFrame(grad_W2, columns=[f"W2_grad_out{j}" for j in range(4)])

result_df = pd.DataFrame.from_dict(result_data, orient="index").T


# 실험 구조를 노드 기반 계산 그래프로 표현
# 구조: Input (5) → Dense(5x3) → Sigmoid(3) → Dense(3x4) → Output(4)

class Node:
    def __init__(self, id, op_type, value=None):
        self.id = id
        self.op_type = op_type
        self.value = value
        self.grad = 0.0
        self.children = []
        self.parents = []

    def add_child(self, child):
        self.children.append(child)
        child.parents.append(self)

# 1. 입력 노드
nodes = []
input_nodes = [Node(i, "input", val) for i, val in enumerate(x.flatten())]
nodes.extend(input_nodes)

# 2. Dense1: mul 노드 (5x3) + sum 노드 (3)
mul_nodes_1 = []
sum_nodes_1 = []
node_id = len(nodes)
for j in range(3):
    muls = []
    for i in range(5):
        mul_node = Node(node_id, "multiply", input_nodes[i].value * W1[i][j])
        mul_node.add_child(input_nodes[i])  # input → mul
        nodes[i].add_child(mul_node)
        muls.append(mul_node)
        nodes.append(mul_node)
        node_id += 1
    # sum node
    sum_node = Node(node_id, "add", sum(m.value for m in muls) + b1[0][j])
    for m in muls:
        sum_node.add_child(m)
        m.add_child(sum_node)
    nodes.append(sum_node)
    sum_nodes_1.append(sum_node)
    node_id += 1

# 3. Activation (Sigmoid)
act_nodes_1 = []
for sum_node in sum_nodes_1:
    sig_val = 1 / (1 + np.exp(-sum_node.value))
    sig_node = Node(node_id, "sigmoid", sig_val)
    sig_node.add_child(sum_node)
    sum_node.add_child(sig_node)
    nodes.append(sig_node)
    act_nodes_1.append(sig_node)
    node_id += 1

# 4. Dense2: mul 노드 (3x4) + sum 노드 (4)
mul_nodes_2 = []
sum_nodes_2 = []
for j in range(4):
    muls = []
    for i in range(3):
        mul_node = Node(node_id, "multiply", act_nodes_1[i].value * W2[i][j])
        mul_node.add_child(act_nodes_1[i])
        act_nodes_1[i].add_child(mul_node)
        muls.append(mul_node)
        nodes.append(mul_node)
        node_id += 1
    sum_node = Node(node_id, "add", sum(m.value for m in muls) + b2[0][j])
    for m in muls:
        sum_node.add_child(m)
        m.add_child(sum_node)
    nodes.append(sum_node)
    sum_nodes_2.append(sum_node)
    node_id += 1

# 5. 최종 출력 노드 (identity 또는 softmax)
output_nodes = []
for s in sum_nodes_2:
    out_node = Node(node_id, "output", s.value)
    out_node.add_child(s)
    s.add_child(out_node)
    nodes.append(out_node)
    output_nodes.append(out_node)
    node_id += 1

# 노드 시각화를 위한 테이블 생성
node_summary = []
for n in nodes:
    node_summary.append({
        "ID": n.id,
        "Op": n.op_type,
        "Value": round(n.value, 6),
        "Parents": [p.id for p in n.parents],
        "Children": [c.id for c in n.children]
    })

node_df = pd.DataFrame(node_summary)

# print(node_df.to_string(index=False))


def convert_node_graph_to_matrix_with_ids(nodes):
    N = len(nodes)
    
    # 1. 연결 행렬
    adj_matrix = np.zeros((N, N), dtype=int)
    for node in nodes:
        for child in node.children:
            adj_matrix[node.id][child.id] = 1
    
    # 2. 연산자 타입
    op_types = [node.op_type for node in nodes]
    
    # 3. 출력 값
    values = np.array([node.value for node in nodes])
    
    # 4. gradient 초기값
    grad_values = np.zeros(N)
    
    # 5. multiply 노드에 대해 input 노드 ID, input 값, weight 값 추적
    params = {}
    for node in nodes:
        if node.op_type == "multiply":
            for p in node.parents:
                if p.op_type in ["input", "sigmoid"]:
                    input_node_id = p.id
                    input_val = p.value
                    weight_val = node.value / input_val if input_val != 0 else 0.0
                    params[node.id] = (input_node_id, input_val, weight_val)
                    break

    return {
        "adj_matrix": adj_matrix,
        "op_types": op_types,
        "values": values,
        "grad_values": grad_values,
        "params": params
    }

def backward_matrix_grad_with_fix(graph_matrix, output_node_ids):
    N = len(graph_matrix["op_types"])
    adj = graph_matrix["adj_matrix"]
    op_types = graph_matrix["op_types"]
    values = graph_matrix["values"]
    grads = graph_matrix["grad_values"].copy()
    params = graph_matrix["params"]

    for i in output_node_ids:
        grads[i] = 1.0

    for i in reversed(range(N)):
        op = op_types[i]
        grad_i = grads[i]
        parent_ids = np.where(adj[:, i])[0]

        if op == "add":
            for pid in parent_ids:
                grads[pid] += grad_i
        elif op == "multiply":
            if i in params:
                input_id, input_val, weight = params[i]
                grads[input_id] += grad_i * weight
        elif op == "sigmoid":
            if len(parent_ids) > 0:
                z_val = values[parent_ids[0]]
                sig_val = values[i]
                grads[parent_ids[0]] += grad_i * sig_val * (1 - sig_val)

    return grads

# 변환 및 재계산
graph_matrix = convert_node_graph_to_matrix_with_ids(nodes)
output_node_ids = [n.id for n in nodes if n.op_type == "output"]
updated_grads = backward_matrix_grad_with_fix(graph_matrix, output_node_ids)
graph_matrix["grad_values"] = updated_grads

# 출력
grad_df = pd.DataFrame({
    "Node ID": list(range(len(updated_grads))),
    "Op Type": graph_matrix["op_types"],
    "Output Value": graph_matrix["values"],
    "Gradient (backprop)": updated_grads
})


print(grad_df.to_string(index=False))