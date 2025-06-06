import cupy as cp
import matrix_ops  # ✅ 변경된 모듈 이름으로 import

# --- 행렬 곱 예제 ---
a = cp.random.rand(64, 128).astype(cp.float32)
b = cp.random.rand(128, 32).astype(cp.float32)
c = cp.zeros((64, 32), dtype=cp.float32)

matrix_ops.matrix_mul(a, b, c, 64, 32, 128)
print(cp.allclose(c, a @ b))  # True

# --- 행렬 덧셈 예제 ---
a = cp.random.rand(1024).astype(cp.float32)
b = cp.random.rand(1024).astype(cp.float32)
c = cp.zeros_like(a)

matrix_ops.matrix_add(a, b, c, a.size)
print(cp.allclose(c, a + b))  # True
