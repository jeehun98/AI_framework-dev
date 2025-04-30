import numpy as np
import scipy.sparse
import pandas as pd

# 실험용 계산 그래프 구성
# 계산식: z = (x + y)^2
# 역전파 시: dz/dx = 2 * (x + y), dz/dy = 2 * (x + y)

# 노드 구성:
# node 0: x
# node 1: y
# node 2: add = x + y
# node 3: square = (x + y)^2

# 1. 인접 행렬 생성 (forward 연결: i → j)
adj_matrix = scipy.sparse.lil_matrix((4, 4))
adj_matrix[0, 2] = 1  # x → add
adj_matrix[1, 2] = 1  # y → add
adj_matrix[2, 3] = 1  # add → square

# 2. 출력값 지정 (순전파 결과)
x_val, y_val = 2.0, 3.0
add_val = x_val + y_val       # 5.0
square_val = add_val ** 2     # 25.0

outputs = np.array([x_val, y_val, add_val, square_val])
grad_output = np.array([0.0, 0.0, 0.0, 1.0])  # 최종 출력에 대한 loss gradient는 1.0

# 3. 연산자 정의 (연산자 index로 구분)
# 0: input (no-op), 1: add, 2: square
op_types = np.array([0, 0, 1, 2])

# 4. 역전파 실행 함수
def backward_pass(adj, ops, outputs, grad_output):
    grads = np.zeros_like(outputs)
    grads[-1] = grad_output[-1]  # 출력 노드에서 시작

    # 뒤에서 앞으로 순서대로 순회
    for i in reversed(range(len(outputs))):
        grad = grads[i]
        if ops[i] == 2:  # square
            # d(square)/d(input) = 2 * input
            input_idx = adj[:, i].nonzero()[0][0]
            grads[input_idx] += 2 * outputs[input_idx] * grad
        elif ops[i] == 1:  # add
            # d(add)/dx = 1
            for j in adj[:, i].nonzero()[0]:
                grads[j] += grad
        # else: input → no-op

    return grads

grads = backward_pass(adj_matrix, op_types, outputs, grad_output)

# 결과를 보기 좋게 출력
df = pd.DataFrame({
    "Node": ["x", "y", "add", "square"],
    "Output": outputs,
    "Grad": grads,
    "Op": ["input", "input", "add", "square"]
})

print("\n[계산 그래프 실험 결과]")
print(df.to_string(index=False))
