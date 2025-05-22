import cupy as cp
import time
import pandas as pd

def benchmark_direct_gpu_allocation(rows, cols, k):
    results = {}

    # 1. 직접 GPU에서 행렬 생성 (복사 비용 없음)
    t0 = time.perf_counter()
    A = cp.random.rand(rows, k).astype(cp.float32)
    B = cp.random.rand(k, cols).astype(cp.float32)
    cp.cuda.Device().synchronize()
    t1 = time.perf_counter()
    results["Direct GPU allocation"] = t1 - t0

    # 2. CuPy 행렬 곱
    t0 = time.perf_counter()
    C = cp.dot(A, B)
    cp.cuda.Device().synchronize()
    t1 = time.perf_counter()
    results["CuPy CUDA compute"] = t1 - t0

    # 3. 결과를 host로 복사
    t0 = time.perf_counter()
    C_host = cp.asnumpy(C)
    t1 = time.perf_counter()
    results["CUDA to NumPy (D2H)"] = t1 - t0

    return pd.DataFrame(list(results.items()), columns=["Stage", "Time (seconds)"])

# 예제 실행
df = benchmark_direct_gpu_allocation(2048, 2048, 2048)
print(df)
