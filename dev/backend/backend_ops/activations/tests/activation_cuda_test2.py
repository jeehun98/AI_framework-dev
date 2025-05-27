import numpy as np
import time

import os
import sys

# CUDA DLL 등록 (Python ≥ 3.8)
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(cuda_path)

# Pybind11 .pyd 경로 추가
pyd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "lib.win-amd64-cpython-312"))
if pyd_path not in sys.path:
    sys.path.insert(0, pyd_path)

import activations_cuda  # ✅ now should succeed


def run_activation_test(sizes, activation="relu"):
    print(f"🚀 테스트 시작 - 활성화 함수: {activation}")
    for size in sizes:
        x = np.random.randn(size).astype(np.float32)
        grad = np.ones_like(x)

        start = time.time()
        out = activations_cuda.apply_activation(x.copy(), activation)
        end = time.time()

        print(f"✅ size={size:7d}, forward time={end - start:.5f} sec")

        # backward test
        start = time.time()
        out_grad = activations_cuda.apply_activation_grad(x.copy(), grad.copy(), activation)
        end = time.time()

        print(f"🔁 backward time={end - start:.5f} sec\n")

if __name__ == "__main__":
    sizes = [1024, 2048, 4096, 1024, 2048, 8192, 4096]  # 재사용 확인용 중복 포함
    run_activation_test(sizes, activation="relu")
