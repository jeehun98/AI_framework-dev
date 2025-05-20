import numpy as np
import pandas as pd

# ------------------------
# 1. 입력 값 생성
# ------------------------
input_values = np.array([1.0, 2.0, 3.0])  # x0, x1, x2, x3

# ------------------------
# 2. 가중치 임의 생성 (4x5)
# ------------------------
np.random.seed(42)
weights = np.random.uniform(0.1, 1.0, size=(3, 2))

# ------------------------
# 3. 노드 및 희소 행렬 정의
# ------------------------
input_nodes = [f"x{i}" for i in range(3)]
mul_nodes = [f"mul_{i}_{j}" for i in range(3) for j in range(2)]
add_nodes = [f"add_{j}" for j in range(2)]
node_labels = input_nodes + mul_nodes + add_nodes
node_to_index = {name: idx for idx, name in enumerate(node_labels)}
N = len(node_labels)

# ------------------------
# 4. 희소 행렬 A 초기화 (노드 연결 관계만 1)
# ------------------------
A = np.zeros((N, N))

# 입력 → 곱셈 노드 연결
for i in range(3):
    for j in range(2):
        A[node_to_index[f"x{i}"], node_to_index[f"mul_{i}_{j}"]] = 1

# 곱셈 노드 → 덧셈 노드 연결
for j in range(2):
    for i in range(3):
        A[node_to_index[f"mul_{i}_{j}"], node_to_index[f"add_{j}"]] = 1

# ------------------------
# 5. 행렬 곱 계산 및 add_j 대각에 삽입
# ------------------------
Z = input_values @ weights  # (1x4) @ (4x5) → (1x5)

for j in range(2):
    add_idx = node_to_index[f"add_{j}"]
    A[add_idx, add_idx] = Z[j]  # add_j 결과를 대각 원소에 삽입

# ------------------------
# 6. 출력
df_final = pd.DataFrame(A, index=node_labels, columns=node_labels)

print(df_final)
print(Z)