# rnn_test.py
import os
import sys
import numpy as np

# CUDA DLL 경로 (Windows)
try:
    os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
except Exception:
    pass

# 프로젝트 경로
ROOT = "C:/Users/owner/Desktop/AI_framework-dev"
sys.path.insert(0, os.path.abspath(ROOT))
sys.path.append(os.path.join(ROOT, "dev", "backend", "graph_executor", "test"))

# 프레임워크 임포트
from dev.models.sequential import Sequential
from dev.layers.Rnn import RNN           # ✅ 단일 RNN 오퍼를 쓰는 RNN 레이어
from dev.layers.activation_layer import Activation
from dev.layers.dense import Dense
import cupy as cp
import graph_executor as ge  # 그래프 확인/역전파 호출용


def make_seq_parity_dataset(B=128, T=12, D=4, seed=0, threshold=None):
    """
    시퀀스 합(첫 번째 피처 기준)이 threshold보다 크면 1, 아니면 0.
    X: (B, T, D) ~ U(0,1)
    y: (B, 1) in {0,1}
    """
    rng = np.random.default_rng(seed)
    x = rng.random((B, T, D), dtype=np.float32)
    # 첫 피처의 합을 기준으로 라벨 생성
    sums = x[:, :, 0].sum(axis=1)
    if threshold is None:
        threshold = np.median(sums)  # 중간값 기준
    y = (sums > threshold).astype(np.float32).reshape(B, 1)
    return x.astype(np.float32), y.astype(np.float32)


def dump_grad_ptr_keys(model, xb, yb, tag):
    """run_graph_backward_entry를 직접 호출하여 grads_ptrs의 키와 shape를 출력."""
    xb_cp = cp.asarray(xb, dtype=cp.float32)
    yb_cp = cp.asarray(yb, dtype=cp.float32)

    # y_true shape 등록(필요 시 1회만)
    model._ensure_label_shape(yb_cp)

    # 텐서 포인터 바인딩
    tensor_ptrs = {"input": xb_cp.data.ptr, "y_true": yb_cp.data.ptr}
    for name, arr in model.weights.items():
        tensor_ptrs[name] = arr.data.ptr
    for name, arr in model.biases.items():
        tensor_ptrs[name] = arr.data.ptr

    grads_ptrs = ge.run_graph_backward_entry(
        E=model.E,
        tensors=tensor_ptrs,
        shapes=model.shapes,
        gradients={},                     # C++에서 채워 반환
        final_output_id=model.output_var, # 손실 이전 출력(=y_pred)
        batch_size=xb_cp.shape[0]
    )

    keys = sorted(grads_ptrs.keys())
    print(f"\n=== grads_ptrs keys ({tag}) ===")
    print("→", keys if keys else "∅ (none)")

    # 각 키의 shape도 같이 출력
    for k in keys:
        shp = model.shapes.get(k, None)
        if shp is None:
            print(f"  - {k}: shape = (unknown)")
        else:
            print(f"  - {k}: shape = ({shp.rows},{shp.cols})")

    # 레퍼런스: 현재 모델이 가진 파라미터 키
    print("\nweights keys:", sorted(model.weights.keys()))
    print("biases  keys:", sorted(model.biases.keys()))
    return grads_ptrs


def print_graph(model):
    """Graph E 노드들을 input/param/output 및 shape까지 보기 좋게 출력"""
    import graph_executor as ge

    print("\n=== [Graph E] ===")
    OPNAME = {
        ge.OpType.MATMUL:     "MATMUL",
        ge.OpType.ADD:        "ADD",
        ge.OpType.RELU:       "RELU",
        ge.OpType.SIGMOID:    "SIGMOID",
        ge.OpType.TANH:       "TANH",
        ge.OpType.FLATTEN:    "FLATTEN",
        ge.OpType.CONV2D:     "CONV2D",
        ge.OpType.LOSS:       "LOSS",
        ge.OpType.LEAKY_RELU: "LEAKY_RELU",
        ge.OpType.ELU:        "ELU",
        ge.OpType.GELU:       "GELU",
        ge.OpType.SILU:       "SILU",
        ge.OpType.SOFTMAX:    "SOFTMAX",
    }
    # 선택적으로 RNN enum이 있을 때만 추가
    if hasattr(ge.OpType, "RNN"):
        OPNAME[ge.OpType.RNN] = "RNN"
    if hasattr(ge.OpType, "POOL_MAX"):
        OPNAME[ge.OpType.POOL_MAX] = "POOL_MAX"
    if hasattr(ge.OpType, "POOL_AVG"):
        OPNAME[ge.OpType.POOL_AVG] = "POOL_AVG"

    def shape_str(shapes, tid):
        s = shapes.get(tid)
        return f"({s.rows},{s.cols})" if s else "(?)"

    for i, op in enumerate(model.E):
        tname = OPNAME.get(op.op_type, str(op.op_type))

        # 벡터/레거시 혼용 대응
        inputs = getattr(op, "inputs", []) or ([op.input_id] if getattr(op, "input_id", "") else [])
        params = getattr(op, "params", []) or ([op.param_id] if getattr(op, "param_id", "") else [])

        in_str  = ", ".join(inputs) if inputs else "-"
        par_str = ", ".join(params) if params else "-"

        # 대표 in/out/param로 shape 표시
        pin  = inputs[0] if inputs else ""
        pout = op.output_id
        sin  = shape_str(model.shapes, pin) if pin else "-"
        sout = shape_str(model.shapes, pout)

        # params는 모두 shape 나열
        spar = ", ".join([f"{pid}:{shape_str(model.shapes, pid)}" for pid in params]) or "-"

        print(f"[{i}] {tname:<9} | in=[{in_str:<20}] {sin:<10} | params=[{spar}] | out={pout:<16} {sout}")

        # ---- extras pretty print ----
        ex = getattr(op, "extra_params", None)
        if ex is not None:
            if hasattr(ge.OpType, "CONV2D") and op.op_type == ge.OpType.CONV2D:
                print(f"     ↳ B={ex.batch_size} Cin={ex.input_c} HxW={ex.input_h}x{ex.input_w} "
                      f"→ Cout={ex.output_c} KhxKw={ex.kernel_h}x{ex.kernel_w} "
                      f"stride={ex.stride_h}x{ex.stride_w} pad={ex.padding_h}x{ex.padding_w}")
            if hasattr(ge.OpType, "POOL_MAX") and op.op_type in (getattr(ge.OpType, "POOL_MAX", -1),
                                                                 getattr(ge.OpType, "POOL_AVG", -1)):
                print(f"     ↳ B={ex.batch_size} C={ex.input_c} HxW={ex.input_h}x{ex.input_w} "
                      f"KhxKw={ex.kernel_h}x{ex.kernel_w} stride={ex.stride_h}x{ex.stride_w} "
                      f"pad={ex.padding_h}x{ex.padding_w} dilation={ex.dilation_h}x{ex.dilation_w} "
                      f"count_include_pad={ex.count_include_pad}")
            if hasattr(ge.OpType, "RNN") and op.op_type == ge.OpType.RNN:
                # 주의: activation 코드를 extra.axis에 임시 저장했었다면 그대로 표시
                print(f"     ↳ B={ex.batch_size} T={ex.time_steps} D={ex.input_w} H={ex.hidden_size} "
                      f"act_code={ex.axis} use_bias={ex.use_bias}")
            if op.op_type == ge.OpType.LOSS:
                print(f"     ↳ loss_type='{ex.loss_type}' label_id='{ex.label_id}'")


def test_rnn_seq_parity():
    print("\n=== [TEST] RNN (single-op) — sequence parity BCE ===")

    # 하이퍼파라미터
    B, T, D = 128, 12, 4
    H = 32  # hidden size

    # 데이터 생성
    x, y = make_seq_parity_dataset(B=B, T=T, D=D, seed=123)

    # 모델 구성 — RNN(H) → Dense(1) → Sigmoid
    #   * RNN 레이어는 내부에서 단일 오퍼(RNN)로 to_e_matrix 구성
    model = Sequential(input_shape=(B, T, D))
    model.add(RNN(units=H, activation=np.tanh, input_shape=(B, T, D), name="rnn",
                  use_backend_init=False))   # 파라미터를 엔진쪽으로 위임(선택)
    model.add(Dense(units=1, activation=None, initializer="xavier", name="fc"))
    model.add(Activation("sigmoid", name="sigm"))

    # 컴파일 (BCE + SGD)
    model.compile(optimizer="sgd", loss="bce", learning_rate=0.05)

    # 그래프 노드 출력
    print_graph(model)

    # 학습 전 손실
    print("\n[BEFORE] evaluate on full set")
    loss_before = model.evaluate(x, y)
    print(f"  BCE(before): {loss_before:.6f}")

    # 🔎 학습 전 grads_ptrs 키 출력
    dump_grad_ptr_keys(model, x[:16], y[:16], tag="before training")

    # 학습
    #  - RNN은 시퀀스 학습 난이도가 있으므로 에폭/러닝레이트는 적당히 조정
    model.fit(x, y, epochs=5000, batch_size=B, verbose=1)

    # 학습 후 손실
    print("\n[AFTER] evaluate on full set")
    loss_after = model.evaluate(x, y)
    print(f"  BCE(after):  {loss_after:.6f}")

    # 🔎 학습 후 grads_ptrs 키 출력
    dump_grad_ptr_keys(model, x[:16], y[:16], tag="after training")

    # 예측 출력 (앞 10개)
    y_pred = model.predict(x)
    print("\n🔍 예측 결과 (앞 10개):")
    print("====================================")
    print("  정답  |  예측값(sigmoid)")
    print("--------|------------------")
    for i in range(min(10, B)):
        print(f"   {int(y[i,0])}   |   {float(y_pred[i,0]):.4f}")
    print("====================================")


if __name__ == "__main__":
    test_rnn_seq_parity()
