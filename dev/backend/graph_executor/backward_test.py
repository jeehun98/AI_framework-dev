import sys
import os
import ctypes

# CUDA DLL 명시적 로드
ctypes.CDLL(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cudart64_12.dll")

# Pybind11 빌드 경로
sys.path.append(os.path.join(os.path.dirname(__file__), "build", "lib.win-amd64-cpython-312"))

import numpy as np
import cupy as cp
import graph_executor as ge

def test_run_graph_backward():
    # 1. 연산 그래프 정의
    E = [
        ge.OpStruct(0, "x", "W", "matmul_out"),     # MATMUL
        ge.OpStruct(1, "matmul_out", "b", "out"),   # ADD
        ge.OpStruct(3, "out", "", "y")              # SIGMOID
    ]

    shapes = {
        "x": ge.Shape(1, 2),
        "W": ge.Shape(2, 2),
        "b": ge.Shape(1, 2),
        "matmul_out": ge.Shape(1, 2),
        "out": ge.Shape(1, 2),
        "y": ge.Shape(1, 2)
    }

    # 2. 입력 및 중간 결과를 GPU 메모리에 할당
    x = cp.array([[1.0, 1.0]], dtype=cp.float32)
    W = cp.array([[0.1, 0.2], [0.3, 0.4]], dtype=cp.float32)
    b = cp.array([[0.0, 0.0]], dtype=cp.float32)

    matmul_out = cp.empty((1, 2), dtype=cp.float32)
    out = cp.empty((1, 2), dtype=cp.float32)
    y = cp.empty((1, 2), dtype=cp.float32)

    tensor_ptrs = {
        "x": x.data.ptr,
        "W": W.data.ptr,
        "b": b.data.ptr,
        "matmul_out": matmul_out.data.ptr,
        "out": out.data.ptr,
        "y": y.data.ptr
    }

    # 3. Forward 실행 (CUDA 내부가 GPU 메모리에 결과 저장하도록 되어 있다고 가정)
    out_host = np.zeros((1, 2), dtype=np.float32)
    ge.run_graph_cuda(E, tensor_ptrs, shapes, out_host, final_output_id="y")
    print("✅ Forward output (host copy):", out_host)

    # 4. Output gradient 설정
    grad_y = cp.array([[1.0, 1.0]], dtype=cp.float32)
    print(f"grad_y ptr: {hex(grad_y.data.ptr)}")
    grad_ptrs = {
        "y": grad_y.data.ptr
    }

    # 5. Backward 실행
    grad_map = ge.run_graph_backward(E, tensor_ptrs, shapes, grad_ptrs, final_output_id="y")
    print("grad_map keys:", grad_map.keys())

    # 6. Gradient 출력 (안전하게 포인터 체크 후 출력)
    def safe_print_grad(name, shape):
        ptr = grad_map.get(name, 0)
        if ptr != 0:
            arr = cp.ndarray(shape, dtype=cp.float32,
                memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(ptr, cp.prod(cp.array(shape)) * 4, None), 0))
            print(f"✅ Gradient {name}:\n", arr.get())
        else:
            print(f"⚠️ Gradient for {name} is NULL (0x0)")

    safe_print_grad("W", (2, 2))
    safe_print_grad("b", (1, 2))
    safe_print_grad("x", (1, 2))

    # 7. 테스트 검증
    assert grad_map["W"] != 0
    assert grad_map["b"] != 0
    assert grad_map["x"] != 0

if __name__ == "__main__":
    test_run_graph_backward()   
    print("끝")
