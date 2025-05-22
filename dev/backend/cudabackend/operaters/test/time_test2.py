import numpy as np
import cupy as cp
import time
import pandas as pd

# Define a larger matrix size
rows, K, cols = 2048, 2048, 2048

# Host (NumPy) matrices
A_np = np.random.rand(rows, K).astype(np.float32)
B_np = np.random.rand(K, cols).astype(np.float32)
C_np = np.zeros((rows, cols), dtype=np.float32)

# Benchmark
stages = []
times = []

# Stage 0: NumPy to CUDA (H2D)
start = time.time()
A_cp = cp.asarray(A_np)
B_cp = cp.asarray(B_np)
C_cp = cp.zeros((rows, cols), dtype=cp.float32)
end = time.time()
stages.append("NumPy to CUDA (H2D)")
times.append(end - start)

# Stage 1: CUDA compute (kernel)
start = time.time()
cp.dot(A_cp, B_cp, out=C_cp)
cp.cuda.Stream.null.synchronize()  # Ensure all operations are complete
end = time.time()
stages.append("CUDA compute (kernel)")
times.append(end - start)

# Stage 2: CUDA to NumPy (D2H)
start = time.time()
C_np = cp.asnumpy(C_cp)
end = time.time()
stages.append("CUDA to NumPy (D2H)")
times.append(end - start)

# Create DataFrame to show results
df = pd.DataFrame({
    "Stage": stages,
    "Time (seconds)": times
})

print(df)