import numpy as np
import pandas as pd

# ------------------------
# 1. 노드 정의
# ------------------------
input_nodes = [f"x{i}" for i in range(4)]
mul_nodes = [f"mul_{i}_{j}" for i in range(4) for j in range(5)]
add_nodes = [f"add_{j}" for j in range(5)]
output_nodes = [f"y{j}" for j in range(5)]

node_labels = input_nodes + mul_nodes + add_nodes + output_nodes
node_to_index = {name: idx for idx, name in enumerate(node_labels)}
N = len(node_labels)

# ------------------------
# 2. 연결 행렬 E (0 또는 전달된 값 저장용)
# ------------------------
E = np.zeros((N, N), dtype=float)

# ------------------------
# 3. 가중치 행렬 W
# ------------------------
W = np.zeros((N, N), dtype=float)
np.random.seed(42)

# 입력 → 곱셈 노드 연결 및 가중치 설정
for i in range(4):
    for j in range(5):
        src = node_to_index[f"x{i}"]
        dst = node_to_index[f"mul_{i}_{j}"]
        E[src][dst] = 0  # 연결 존재
        W[src][dst] = np.random.uniform(0.1, 1.0)

# 곱셈 → 덧셈 노드 연결
for j in range(5):
    for i in range(4):
        src = node_to_index[f"mul_{i}_{j}"]
        dst = node_to_index[f"add_{j}"]
        E[src][dst] = 0
        W[src][dst] = 1.0

# 덧셈 → 출력 노드 연결
for j in range(5):
    src = node_to_index[f"add_{j}"]
    dst = node_to_index[f"y{j}"]
    E[src][dst] = 0
    W[src][dst] = 1.0

# ------------------------
# 4. 입력값 초기화 (E에 직접 삽입)
# ------------------------
input_values = [1.0, 2.0, 3.0, 4.0]
for i in range(4):
    for j in range(5):
        src = node_to_index[f"x{i}"]
        dst = node_to_index[f"mul_{i}_{j}"]
        E[src][dst] = input_values[i] * W[src][dst]

# ------------------------
# 5. 연산 수행 (E에 직접 덮어쓰기)
# ------------------------
for j in range(N):  # 수신 노드
    # 노드 j로 들어오는 값 합산
    incoming_sum = np.sum(E[:, j])
    if incoming_sum == 0:
        continue

    # 노드 j에서 나가는 노드 k로 전달
    for k in range(N):
        if W[j][k] != 0:
            E[j][k] = incoming_sum * W[j][k]

# ------------------------
# 6. 결과 출력
# ------------------------
df_E_result = pd.DataFrame(E, columns=node_labels, index=node_labels)

# E 행렬에서 0이 아닌 값들만 추출
nonzero_entries = []

for i in range(N):
    for j in range(N):
        if E[i][j] != 0:
            from_node = node_labels[i]
            to_node = node_labels[j]
            value = E[i][j]
            nonzero_entries.append((from_node, to_node, value))

# DataFrame으로 정리
df_nonzero = pd.DataFrame(nonzero_entries, columns=["From", "To", "Value"])

print(df_nonzero)

# 입력값 x_i 복원 검증용 코드
# mul 입력: x_i * w_ij = E[i][j]
# 역으로: x_i = E[i][j] / W[i][j]

recovered_inputs = {}

for i in range(4):  # 입력 노드: x0 ~ x3
    estimates = []
    for j in range(5):
        src = node_to_index[f"x{i}"]
        dst = node_to_index[f"mul_{i}_{j}"]
        w = W[src][dst]
        if w != 0:
            x_est = E[src][dst] / w
            estimates.append(x_est)
    if estimates:
        avg_est = sum(estimates) / len(estimates)
        recovered_inputs[f"x{i}"] = round(avg_est, 6)  # 평균으로 대표값 추정

# 비교용: 원래 입력값
original_inputs = {f"x{i}": val for i, val in enumerate([1.0, 2.0, 3.0, 4.0])}

# 결과 정리
df_verify = pd.DataFrame({
    "Original x_i": original_inputs,
    "Recovered x_i": recovered_inputs
})

print(df_verify)