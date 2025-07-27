import sys
import os
import ctypes

# CUDA DLL 로드
ctypes.CDLL(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cudart64_12.dll")

# Pybind11 빌드 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "build", "lib.win-amd64-cpython-312"))

import cupy as cp
import numpy as np
from graph_executor import OpStruct, Shape, run_graph_cuda, run_graph_backward

cp.cuda.Device(0).use()  # ✅ CUDA 디바이스 명시

batch_size = 2

# ✅ 입력 샘플 2개
x = cp.array([[1.0, 2.0],
              [2.0, 3.0]], dtype=cp.float32)
W = cp.array([[1.0, 0.0],
              [0.0, 1.0]], dtype=cp.float32)
b = cp.array([[0.5, -0.5]], dtype=cp.float32)

# ✅ 계산 그래프 정의
E = [
    OpStruct(0, "x0", "W", "linear"),
    OpStruct(1, "linear", "b", "out"),
    OpStruct(3, "out", "", "act_out"),
]

# ✅ 텐서 shape 정의 (모두 batch 단위로 설정)
shapes = {
    "x0": Shape(batch_size, 2),
    "W": Shape(2, 2),
    "b": Shape(1, 2),
    "linear": Shape(batch_size, 2),
    "out": Shape(batch_size, 2),
    "act_out": Shape(batch_size, 2),
}

# ✅ 중간 결과 버퍼 등록 (CUDA 내부 연산을 위해 포인터 필요)
linear_buf = cp.empty((batch_size, 2), dtype=cp.float32)
out_buf = cp.empty((batch_size, 2), dtype=cp.float32)
act_out_buf = cp.empty((batch_size, 2), dtype=cp.float32)

# ✅ 텐서 포인터 정의 (입력 + 중간 결과)
tensors = {
    "x0": int(x.data.ptr),
    "W": int(W.data.ptr),
    "b": int(b.data.ptr),
    "linear": int(linear_buf.data.ptr),
    "out": int(out_buf.data.ptr),
    "act_out": int(act_out_buf.data.ptr),
}

# ✅ Forward 결과 저장 버퍼 (host로 복사)
out_host = np.zeros((batch_size, 2), dtype=np.float32)

# ✅ Forward 실행
run_graph_cuda(E, tensors, shapes, out_host, final_output_id="act_out", batch_size=batch_size)

print("✅ forward output:")
print(out_host)

# ✅ 역전파 입력: grad(act_out)
grad_act_out = cp.ones((batch_size, 2), dtype=cp.float32)
print("데이터 확인", grad_act_out.data.ptr, grad_act_out)

gradient_ptrs = {
    "act_out": int(grad_act_out.data.ptr)
}

# ✅ Backward 실행
grad_result = run_graph_backward(
    E=E,
    tensors=tensors,
    shapes=shapes,
    gradients=gradient_ptrs,
    final_output_id="act_out",
    batch_size=batch_size
)

# ✅ 결과 출력
print("\n✅ 반환된 gradient 포인터:")
for name, ptr in grad_result.items():
    print(f"{name}: {ptr}")

print("\n✅ 역전파 결과 (gradient 내용):")
    
# 🔹 출력 우선순위: W, b, x0, 그 외
preferred_order = ["W", "b", "x0", "linear", "out", "act_out"]

for name in preferred_order + [k for k in grad_result.keys() if k not in preferred_order]:
    ptr = grad_result.get(name, 0)
    if ptr == 0:
        print(f"[WARNING] {name} returned null pointer. Skipping.")
        continue

    shape = shapes.get(name)
    if shape is None:
        print(f"[WARNING] Shape not found for {name}. Skipping.")
        continue

    size = shape.rows * shape.cols
    grad_np = cp.ndarray((shape.rows, shape.cols), dtype=cp.float32,
                         memptr=cp.cuda.MemoryPointer(
                             cp.cuda.UnownedMemory(ptr, size * 4, 0), 0))
    print(f"{name}:\n{grad_np.get()}")
