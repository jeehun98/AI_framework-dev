import numpy as np
import pandas as pd

# ------------------------
# 1. 노드 정의 (activation 포함)
# ------------------------
input_nodes = [f"x{i}" for i in range(4)]
mul_nodes = [f"mul_{i}_{j}" for i in range(4) for j in range(5)]
add_nodes = [f"add_{j}" for j in range(5)]
act_nodes = [f"act_{j}" for j in range(5)]
output_nodes = [f"y{j}" for j in range(5)]

node_labels = input_nodes + mul_nodes + add_nodes + act_nodes + output_nodes
node_to_index = {name: idx for idx, name in enumerate(node_labels)}
N = len(node_labels)

# ------------------------
# 2. 행렬 E, W 초기화
# ------------------------
E = np.zeros((N, N), dtype=float)
W = np.zeros((N, N), dtype=float)
np.random.seed(42)

# ------------------------
# 3. 연결 구성 및 가중치 초기화
# ------------------------

# 입력 → 곱셈
for i in range(4):
    for j in range(5):
        src = node_to_index[f"x{i}"]
        dst = node_to_index[f"mul_{i}_{j}"]
        W[src][dst] = np.random.uniform(0.1, 1.0)

# 곱셈 → 덧셈
for j in range(5):
    for i in range(4):
        src = node_to_index[f"mul_{i}_{j}"]
        dst = node_to_index[f"add_{j}"]
        W[src][dst] = 1.0

# 덧셈 → 활성화
for j in range(5):
    src = node_to_index[f"add_{j}"]
    dst = node_to_index[f"act_{j}"]
    W[src][dst] = 1.0

# 활성화 → 출력
for j in range(5):
    src = node_to_index[f"act_{j}"]
    dst = node_to_index[f"y{j}"]
    W[src][dst] = 1.0

# ------------------------
# 4. 입력값 설정 및 초기 E 구성
# ------------------------
input_values = [1.0, 2.0, 3.0, 4.0]
for i in range(4):
    for j in range(5):
        src = node_to_index[f"x{i}"]
        dst = node_to_index[f"mul_{i}_{j}"]
        E[src][dst] = input_values[i] * W[src][dst]

# ------------------------
# 5. 연산 함수 정의
# ------------------------

def relu(x):
    return np.maximum(0, x)

def forward_step(E, W):
    V_out = np.sum(E, axis=0)  # 노드 출력값
    E_tiled = np.tile(V_out.reshape(N, 1), (1, N))  # 타일링
    E_new = E_tiled * W  # 전달값 계산

    # 덧셈 → 활성화 노드
    for j in range(5):
        add_idx = node_to_index[f"add_{j}"]
        act_idx = node_to_index[f"act_{j}"]
        add_output = np.sum(E_new[:, add_idx])
        relu_output = relu(add_output)
        E_new[add_idx][act_idx] = relu_output * W[add_idx][act_idx]

    # 활성화 → 출력 노드
    for j in range(5):
        act_idx = node_to_index[f"act_{j}"]
        y_idx = node_to_index[f"y{j}"]
        act_output = np.sum(E_new[:, act_idx])
        E_new[act_idx][y_idx] = act_output * W[act_idx][y_idx]

    return E_new

# ------------------------
# 6. forward 실행 및 출력
# ------------------------
E_result = forward_step(E, W)

# 0이 아닌 값만 추출
nonzero_entries = []
for i in range(N):
    for j in range(N):
        if E_result[i][j] != 0:
            nonzero_entries.append((node_labels[i], node_labels[j], E_result[i][j]))

df_E_result = pd.DataFrame(E_result, columns=node_labels, index=node_labels)
df_nonzero = pd.DataFrame(nonzero_entries, columns=["From", "To", "Value"])

print(df_nonzero)