import numpy as np
import pandas as pd

# 4 입력 → 5 출력 fully connected
# 각 y_j는 4개의 mul 노드를 모두 직접 받아 1개의 add 노드로 합산

node_labels = []
edges = []

# 1. 입력 노드
input_nodes = [f"x{i}" for i in range(4)]
node_labels.extend(input_nodes)

# 2. 곱셈 노드 (x_i * w_ij)
multiply_nodes = []
for i in range(4):
    for j in range(5):
        mul_node = f"mul_{i}_{j}"
        multiply_nodes.append(mul_node)
        node_labels.append(mul_node)
        edges.append((f"x{i}", mul_node))

# 3. add 노드 (각 출력당 1개만)
add_nodes = []
output_nodes = []
for j in range(5):
    add_node = f"add_{j}"
    add_nodes.append(add_node)
    node_labels.append(add_node)

    # 연결: mul_0_j, ..., mul_3_j → add_j
    for i in range(4):
        edges.append((f"mul_{i}_{j}", add_node))

    # 출력 노드
    y_node = f"y{j}"
    output_nodes.append(y_node)
    node_labels.append(y_node)
    edges.append((add_node, y_node))

# 4. 연결 행렬 생성
node_to_index = {label: idx for idx, label in enumerate(node_labels)}
N = len(node_labels)
adj_matrix = np.zeros((N, N), dtype=int)

for src, dst in edges:
    i, j = node_to_index[src], node_to_index[dst]
    adj_matrix[i][j] = 1

# DataFrame으로 출력
cols = [f"{label}" for label in node_labels]
df_adj = pd.DataFrame(adj_matrix, columns=cols, index=cols)

# 연결만 요약 출력
nonzero_edges = [(src, dst) for src, dst in edges]
df_edges = pd.DataFrame(nonzero_edges, columns=["From", "To"])

print(df_edges)