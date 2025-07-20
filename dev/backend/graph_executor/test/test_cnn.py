import sys
import os
import ctypes
import numpy as np
import cupy as cp

# CUDA DLL 명시적 로드 (필요 시)
ctypes.CDLL(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cudart64_12.dll")

# Pybind11로 빌드된 .pyd 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "build", "lib.win-amd64-cpython-312"))

import graph_executor as ge

# ======= 1. 입력 및 파라미터 정의 =======
B, H, W = 1, 5, 5           # Input: (B, H, W)
KH, KW = 3, 3               # Kernel: (KH, KW)
OH = H - KH + 1             # Output Height (stride=1, padding=0)
OW = W - KW + 1             # Output Width
print(f"[INFO] Output shape: ({B}, {OH}, {OW})")

# 입력 데이터
x_np = np.random.rand(B, H, W).astype(np.float32)
K_np = np.ones((KH, KW), dtype=np.float32)  # 간단한 평균 필터

# GPU로 전송
x = cp.asarray(x_np)
K = cp.asarray(K_np)
out = cp.zeros((B, OH, OW), dtype=cp.float32)

x_ptr = x.data.ptr
K_ptr = K.data.ptr
out_ptr = out.data.ptr

# ======= 2. Shape 및 Tensor Mapping =======
shapes = {
    "x": ge.Shape(B * H, W),
    "K": ge.Shape(KH, KW),
    "out": ge.Shape(OH, OW)
}

tensors = {
    "x": x_ptr,
    "K": K_ptr,
    "out": out_ptr
}

# ======= 3. 연산 노드 정의 =======
extra = ge.OpExtraParams()
extra.kernel_h = KH
extra.kernel_w = KW
extra.input_h = H
extra.input_w = W
extra.batch_size = B

op = ge.OpStruct(ge.OpType.CONV2D, "x", "K", "out", extra)
E = [op]

# ======= 4. Forward 실행 =======
ge.run_graph_cuda(E, tensors, shapes, out_ptr, "out", B)
print("✅ Forward 결과:\n", cp.asnumpy(out))

# ======= 5. Backward 실행 =======
grad_ptrs = {}
grads = ge.run_graph_backward(E, tensors, shapes, grad_ptrs, "out", B)

# ======= 6. Gradient 결과 출력 =======
def copy_gpu_ptr(ptr, shape):
    size = np.prod(shape)
    mem = cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(ptr, size * 4, None), 0)
    return cp.ndarray(shape, dtype=cp.float32, memptr=mem)

if "x" in grads:
    dx = copy_gpu_ptr(grads["x"], (H, W))
    print("dL/dx:\n", cp.asnumpy(dx))
else:
    print("[ERROR] Gradient for 'x' not found!")

if "K" in grads:
    dK = copy_gpu_ptr(grads["K"], (KH, KW))
    print("dL/dK:\n", cp.asnumpy(dK))
else:
    print("[ERROR] Gradient for 'K' not found!")
