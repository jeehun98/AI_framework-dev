import numpy as np
import cupy as cp
import time
import pandas as pd

# 실험: 직접 GPU 메모리에 생성된 cupy 배열을 활용한 경우
def benchmark_direct_cupy_matrix(rows, cols, k):
    times = {}

    # 1. GPU에서 직접 생성 (따라서 H2D 필요 없음)
    t0 = time.perf_counter()
    A = cp.random.rand(rows, k).astype(cp.float32)
    B = cp.random.rand(k, cols).astype(cp.float32)
    cp.cuda.Device().synchronize()
    t1 = time.perf_counter()
    times["Create directly on CUDA"] = t1 - t0

    # 2. GPU 계산
    t0 = time.perf_counter()
    C = cp.dot(A, B)
    cp.cuda.Device().synchronize()
    t1 = time.perf_counter()
    times["CUDA compute (cupy)"] = t1 - t0

    # 3. GPU → NumPy로 복사 (D2H)
    t0 = time.perf_counter()
    C_host = cp.asnumpy(C)
    t1 = time.perf_counter()
    times["CUDA to NumPy (D2H)"] = t1 - t0

    return pd.DataFrame(list(times.items()), columns=["Stage", "Time (seconds)"])

df_result = benchmark_direct_cupy_matrix(2048, 2048, 2048)

print(df_result)