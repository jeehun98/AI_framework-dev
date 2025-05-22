import numpy as np
import cupy as cp
import torch
import time
import pandas as pd

timing = {}

shape = (4096, 4096)

# ▶️ NumPy 생성 후 GPU로 복사 (CuPy)
start = time.time()
A_np = np.random.rand(*shape).astype(np.float32)
B_np = np.random.rand(*shape).astype(np.float32)
timing["NumPy creation"] = time.time() - start

start = time.time()
A_cp = cp.asarray(A_np)
B_cp = cp.asarray(B_np)
timing["H2D copy (NumPy → CuPy)"] = time.time() - start

start = time.time()
C_cp = A_cp @ B_cp
cp.cuda.Device(0).synchronize()
timing["CuPy matmul (after H2D)"] = time.time() - start

# ▶️ CuPy 직접 생성
start = time.time()
A_gpu = cp.random.rand(*shape, dtype=cp.float32)
B_gpu = cp.random.rand(*shape, dtype=cp.float32)
timing["Direct CuPy creation"] = time.time() - start

start = time.time()
C_gpu = A_gpu @ B_gpu
cp.cuda.Device(0).synchronize()
timing["CuPy matmul (direct)"] = time.time() - start

# ▶️ PyTorch: CPU 생성 후 GPU 복사
start = time.time()
A_torch_cpu = torch.rand(*shape, dtype=torch.float32)
B_torch_cpu = torch.rand(*shape, dtype=torch.float32)
timing["PyTorch CPU tensor creation"] = time.time() - start

start = time.time()
A_torch = A_torch_cpu.cuda()
B_torch = B_torch_cpu.cuda()
torch.cuda.synchronize()
timing["H2D copy (Torch → GPU)"] = time.time() - start

start = time.time()
C_torch = A_torch @ B_torch
torch.cuda.synchronize()
timing["Torch matmul (after H2D)"] = time.time() - start

# ▶️ PyTorch: GPU 직접 생성
start = time.time()
A_torch_gpu = torch.rand(*shape, dtype=torch.float32, device="cuda")
B_torch_gpu = torch.rand(*shape, dtype=torch.float32, device="cuda")
torch.cuda.synchronize()
timing["Direct Torch CUDA tensor creation"] = time.time() - start

start = time.time()
C_torch_gpu = A_torch_gpu @ B_torch_gpu
torch.cuda.synchronize()
timing["Torch matmul (direct)"] = time.time() - start

# ▶️ 결과 출력
df = pd.DataFrame(list(timing.items()), columns=["Stage", "Time (seconds)"])

print(df)