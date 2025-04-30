import numpy as np
import pandas as pd

# 입력 데이터 (1, 3)
A = [[1.0, 2.0, 3.0]]  # shape (1, 3)

# 가중치 행렬 (3, 4)
B = [
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 1.0, 1.1, 1.2],
]  # shape (3, 4)

# 행렬 곱 결과 = A @ B = shape (1, 4)
result = np.dot(A, B).tolist()

# 계산 그래프 노드 구조 시뮬레이션
# 각 sum_node (출력 유닛)당 mul_node (입력 차원 수만큼)

node_records = []

for i in range(1):  # batch size = 1
    for j in range(4):  # output units
        sum_output = result[i][j]
        sum_record = {
            "Node": f"sum_{i}_{j}",
            "Type": "add",
            "Output": sum_output,
            "Input": None,
            "Weight": None,
            "Child of": None
        }
        node_records.append(sum_record)

        for k in range(3):  # input dimension
            mul_output = A[i][k] * B[k][j]
            mul_record = {
                "Node": f"mul_{i}_{j}_{k}",
                "Type": "multiply",
                "Output": mul_output,
                "Input": A[i][k],
                "Weight": B[k][j],
                "Child of": f"sum_{i}_{j}"
            }
            node_records.append(mul_record)

df = pd.DataFrame(node_records)

print("\n[계산 그래프 실험 결과]")
print(df.to_string(index=False))

# 역전파 수행
# forward 계산: z_j = sum_i (A_i * B_ij)
# backward: dL/dA_i = sum_j (dL/dz_j * B_ij)
#            dL/dB_ij = A_i * dL/dz_j

# grad_output: dL/dz_j, 출력 노드에 대한 gradient (1, 4)
grad_output = np.array([[1.0, 1.0, 1.0, 1.0]])  # 모든 출력에 대한 loss gradient = 1.0

A_np = np.array(A)  # (1, 3)
B_np = np.array(B)  # (3, 4)

# dL/dA = grad_output @ B.T   → shape: (1, 3)
grad_input = np.dot(grad_output, B_np.T)

# dL/dB = A.T @ grad_output   → shape: (3, 4)
grad_weights = np.dot(A_np.T, grad_output)

# 결과 정리
grad_A_df = pd.DataFrame(grad_input, columns=[f"dL/dA_{i}" for i in range(3)])
grad_B_df = pd.DataFrame(grad_weights, columns=[f"dL/dB_{j}" for j in range(4)])

print(grad_A_df.to_string(index=False))
print(grad_B_df.to_string(index=False))