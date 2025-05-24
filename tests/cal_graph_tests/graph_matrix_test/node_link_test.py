import numpy as np
import pandas as pd

N = 4

E = np.array([
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
], dtype=float)

W = np.array([
    [0, 0, 0.5, 0],
    [0, 0, 0.8, 0],
    [0, 0, 0, 1.2],
    [0, 0, 0, 0],
], dtype=float)

# 입력값 설정
E[0][2] = 1.0
E[1][2] = 2.0

def custom_forward(E, W):
    N = E.shape[0]
    E_next = E.copy()

    for j in range(N):
        acc = 0.0
        has_valid_input = False

        for i in range(N):
            if E[i][j] != 0 and W[i][j] != 0:
                acc += E[i][j] * W[i][j]
                has_valid_input = True

        for k in range(N):
            if E[j][k] == 1 and has_valid_input:
                E_next[j][k] = acc  # 결과 overwrite
                # ✅ 중요: 입력값이 있는 위치는 건드리지 않음

    return E_next

E_after = custom_forward(E, W)

# 출력
df_E_before = pd.DataFrame(E, columns=[f"N{j}" for j in range(N)], index=[f"N{i}" for i in range(N)])
df_E_after = pd.DataFrame(E_after, columns=[f"N{j}" for j in range(N)], index=[f"N{i}" for i in range(N)])
df_W = pd.DataFrame(W, columns=[f"N{j}" for j in range(N)], index=[f"N{i}" for i in range(N)])

print("연산 전 E 행렬:")
print(df_E_before)

print("\n가중치 W 행렬:")
print(df_W)

print("\n1회 forward 후 E 행렬:")
print(df_E_after)
