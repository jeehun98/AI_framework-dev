import sys
import os
import numpy as np
import cupy as cp
import ctypes

# CUDA DLL 명시적 로드
ctypes.CDLL(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cudart64_12.dll")

# Pybind11 빌드된 .pyd 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "../build/lib.win-amd64-cpython-312"))
import graph_executor as ge

def test_run_graph_with_loss():
    # ✅ 테스트용 simple Dense 그래프 구성
    E = [
        ge.OpStruct(ge.OpType.MATMUL, "x", "w", "z", ge.OpExtraParams()),
        ge.OpStruct(ge.OpType.ADD, "z", "b", "a", ge.OpExtraParams()),
        ge.OpStruct(ge.OpType.SIGMOID, "a", "", "y_pred", ge.OpExtraParams()),
        ge.OpStruct(ge.OpType.LOSS, "y_pred", "", "loss_out", ge.OpExtraParams())
    ]

    # ✅ 텐서 정의
    batch_size = 1
    input_dim = 2
    output_dim = 1

    x = cp.array([[0.5, 1.0]], dtype=cp.float32)
    w = cp.array([[0.3], [0.7]], dtype=cp.float32)
    b = cp.array([[0.1]], dtype=cp.float32)
    y_true = cp.array([[1.0]], dtype=cp.float32)
    y_pred = cp.zeros((1, 1), dtype=cp.float32)  # ✅ 예측값 출력 버퍼

    # ✅ 텐서 및 shape 등록
    tensors = {
        "x": x,
        "w": w,
        "b": b,
        "y_true": y_true,
        "y_pred": y_pred,  # ✅ 연산 출력 등록
    }

    shapes = {
        "x": ge.Shape(1, 2),
        "w": ge.Shape(2, 1),
        "b": ge.Shape(1, 1),
        "y_true": ge.Shape(1, 1),
        "y_pred": ge.Shape(1, 1),  # ✅ 출력 shape 추가
    }

    # ✅ 텐서 포인터 변환
    tensor_ptrs = {k: tensor.data.ptr for k, tensor in tensors.items()}

    loss_type = "mse"

    # ✅ run_graph_with_loss 호출 (이전에 y_pred 메모리 등록 완료됨)
    loss_val = ge.run_graph_with_loss_entry(
        E=E,
        tensors=tensor_ptrs,
        shapes=shapes,
        final_output_id="y_pred",
        label_tensor_id="y_true",
        loss_type=loss_type,
        batch_size=batch_size
    )

    cp.cuda.Device().synchronize()  # ✅ 안전한 결과 확인을 위한 동기화

    print("=== [TEST] run_graph_with_loss_entry ===")
    print(f"Loss Type     : {loss_type}")
    print(f"Predicted y   : {cp.asnumpy(y_pred)}")   # ✅ 직접 버퍼 확인
    print(f"Ground Truth  : {cp.asnumpy(y_true)}")
    print(f"Computed Loss : {loss_val:.6f}")

if __name__ == "__main__":
    test_run_graph_with_loss()
