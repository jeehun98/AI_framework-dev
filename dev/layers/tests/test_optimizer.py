# test_optimizer_with_wiring_debug.py
import os
import sys
import numpy as np
import cupy as cp

# ===== CUDA DLL 경로 (Windows, 필요 시 조정) =====
try:
    os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
except Exception:
    pass

# ===== 프로젝트/바인딩 모듈 경로 =====
# graph_executor .pyd 경로
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend", "graph_executor", "build", "lib.win-amd64-cpython-312"))
# 프로젝트 루트
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev"))

# ===== 프레임워크 임포트 =====
from dev.models.sequential import Sequential
from dev.layers.dense import Dense
from dev.layers.activation_layer import Activation
from dev.layers.flatten import Flatten
import graph_executor as ge  # wiring 점검에 사용


# ========== numeric vs backprop (W 그라디언트 부호 체크) ==========
def numeric_vs_backprop_grad_W(model, x, y, eps=1e-3):
    """
    단일 Dense(회귀) 가정에서 W의 dL/dW 부호를 수치미분 vs 역전파로 비교.
    - 기본은 (1,1) W 가정. (더 큰 경우 ih,iw 바꿔 반복)
    """
    # 1) 대상 W 파라미터 이름 추출(첫 번째 *_W)
    wname = None
    for k in model.weights.keys():
        if k.endswith("_W"):
            wname = k
            break
    if wname is None:
        raise RuntimeError("No weight param (*_W) found on model.")

    W = model.weights[wname]  # CuPy array
    W_np = cp.asnumpy(W)
    ih, iw = 0, 0
    if W_np.ndim != 2:
        raise RuntimeError(f"{wname} must be 2D; got {W_np.shape}")

    # 2) 원래 손실
    loss0 = float(model.evaluate(x, y))

    # 3) 수치미분 (central difference)
    orig = float(W_np[ih, iw])
    W[ih, iw] = orig + eps
    loss_plus  = float(model.evaluate(x, y))
    W[ih, iw] = orig - eps
    loss_minus = float(model.evaluate(x, y))
    W[ih, iw] = orig  # 복원
    num_grad = (loss_plus - loss_minus) / (2 * eps)

    # 4) 역전파 그라디언트 계산 (run_graph_backward_entry)
    x_cp = cp.asarray(x, dtype=cp.float32)
    y_cp = cp.asarray(y, dtype=cp.float32)
    tensor_ptrs = {"input": x_cp.data.ptr, "y_true": y_cp.data.ptr}
    for name, arr in model.weights.items():
        tensor_ptrs[name] = arr.data.ptr
    for name, arr in model.biases.items():
        tensor_ptrs[name] = arr.data.ptr

    grads_ptrs = {}
    grads_dict = ge.run_graph_backward_entry(
        E=model.E,
        tensors=tensor_ptrs,
        shapes=model.shapes,
        gradients=grads_ptrs,
        final_output_id=(getattr(model, "loss_output_id", None) or model.output_var),
        batch_size=x.shape[0]
    )

    # 5) W grad 포인터를 CuPy ndarray 로 래핑 후 값 읽기
    shp = model.shapes[wname]
    Kh, Kw = int(shp.rows), int(shp.cols)
    wgrad_ptr = int(grads_dict[wname])  # uintptr
    mem = cp.cuda.UnownedMemory(wgrad_ptr, Kh * Kw * 4, model)  # owner로 model을 넣어 참조 유지
    mp  = cp.cuda.MemoryPointer(mem, 0)
    wgrad_cp = cp.ndarray((Kh, Kw), dtype=cp.float32, memptr=mp)
    backprop_grad = float(cp.asnumpy(wgrad_cp)[ih, iw])

    same_sign = np.sign(num_grad) == np.sign(backprop_grad)
    print(f"[GradCheck W] numeric: {num_grad:.6e} | backprop: {backprop_grad:.6e} | same sign? {same_sign}")


# ========== 유틸: CuPy/NumPy 안전 변환 ==========
def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    try:
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    if hasattr(x, "get"):
        try:
            return x.get()
        except Exception:
            pass
    if hasattr(x, "cpu"):
        try:
            return x.cpu().numpy()
        except Exception:
            pass
    return np.asarray(x)

def flatten_params(params_dict):
    flats = []
    for k in sorted(params_dict.keys()):
        arr_np = to_numpy(params_dict[k]).astype(np.float32, copy=False)
        flats.append(arr_np.ravel())
    if flats:
        return np.concatenate(flats)
    return np.zeros(0, dtype=np.float32)

def dot(delta, grad):
    delta = to_numpy(delta).astype(np.float32, copy=False).ravel()
    grad  = to_numpy(grad).astype(np.float32, copy=False).ravel()
    if delta.size == 0 or grad.size == 0:
        return 0.0
    return float(np.dot(delta, grad))

def is_bad(x):
    x = to_numpy(x)
    return not np.isfinite(x).all()

def l2norm(x):
    x = to_numpy(x).astype(np.float32, copy=False).ravel()
    return float(np.linalg.norm(x) if x.size else 0.0)

def get_params(model):
    params = {}
    params.update(getattr(model, "weights", {}))
    params.update(getattr(model, "biases", {}))
    return params

def dump_param_stats(model, tag=""):
    params = get_params(model)
    stats = {k: (l2norm(v), float(np.min(to_numpy(v))), float(np.max(to_numpy(v))))
             for k, v in params.items()}
    print(f"[{tag}] param stats (||·||2, min, max): {stats}")


# ========== 그래프↔파라미터 배선 점검 ==========
def debug_graph_bindings(model):
    uses_param = {ge.OpType.MATMUL, ge.OpType.ADD, ge.OpType.CONV2D}
    graph_param_ids = sorted({op.param_id for op in model.E if (op.op_type in uses_param) and op.param_id})
    weight_keys = sorted(getattr(model, "weights", {}).keys())
    bias_keys   = sorted(getattr(model, "biases", {}).keys())
    shape_keys  = sorted(getattr(model, "shapes", {}).keys())
    tensor_keys = sorted(set(weight_keys) | set(bias_keys))

    print("\n==== Graph/Param wiring check ====")
    print("Graph param_ids:", graph_param_ids)
    print("weights keys:", weight_keys)
    print("biases  keys:", bias_keys)
    print("shapes  keys (first 20):", shape_keys[:20])
    print("tensor keys:", tensor_keys)

    miss_tensors = [k for k in graph_param_ids if k not in tensor_keys]
    miss_shapes  = [k for k in graph_param_ids if k not in shape_keys]
    if miss_tensors:
        print("⚠️  MISSING in tensors:", miss_tensors)
    if miss_shapes:
        print("⚠️  MISSING in shapes:", miss_shapes)
    if not graph_param_ids:
        print("❌ Graph has NO trainable params (no MATMUL/ADD/CONV2D with param_id).")
    elif not tensor_keys:
        print("❌ Model has NO param buffers in weights/biases.")
    elif not miss_tensors and not miss_shapes:
        print("✅ Wiring looks good.")


# ========== 모델 빌더 ==========
def build_xor_model(lr, optimizer):
    model = Sequential(input_shape=(1, 1, 2))
    model.add(Flatten(input_shape=(1, 1, 2)))
    model.add(Dense(units=4, activation=None, initializer="xavier"))
    model.add(Activation("tanh"))
    model.add(Dense(units=1, activation=None, initializer="xavier"))
    model.add(Activation("sigmoid"))
    model.compile(optimizer=optimizer, loss="bce", learning_rate=lr)
    return model

def build_reg_model(lr, optimizer):
    # y = 3x - 1 (MSE)
    model = Sequential(input_shape=(1, 1, 1))
    model.add(Flatten(input_shape=(1, 1, 1)))
    model.add(Dense(units=1, activation=None, initializer="xavier"))
    model.compile(optimizer=optimizer, loss="mse", learning_rate=lr)
    return model


# ========== 테스트 ==========
def test_optimizer_on_xor(opt_name, lr=0.3, epochs=1000):
    print(f"\n=== XOR / {opt_name} ===")
    np.random.seed(42)

    x = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32).reshape(4,1,1,2)
    y = np.array([[0],[1],[1],[0]], dtype=np.float32)

    model = build_xor_model(lr=lr, optimizer=opt_name)
    debug_graph_bindings(model)

    loss_before = float(model.evaluate(x, y))
    print(f"  BCE(before): {loss_before:.6f}")
    dump_param_stats(model, tag="init")

    params_before = flatten_params(get_params(model))
    model.fit(x, y, epochs=1, batch_size=len(x))
    params_after_1 = flatten_params(get_params(model))
    delta_1 = params_after_1 - params_before

    try:
        grads = getattr(model, "last_grads", None)
        if grads is not None:
            grads_flat = flatten_params(grads)
            grad_inner = dot(delta_1, grads_flat)
            print(f"  ΔW·grad(after 1 step): {grad_inner:.6e} (expected < 0)")
    except Exception:
        pass

    remain = max(0, epochs-1)
    for e in range(remain):
        model.fit(x, y, epochs=1, batch_size=len(x))
        if (e+1) % 100 == 0:
            mid = float(model.evaluate(x, y))
            print(f"  [epoch+{e+1}] BCE={mid:.6f}")
            if not np.isfinite(mid):
                raise RuntimeError(f"BCE became non-finite at epoch {e+1}")

    loss_after = float(model.evaluate(x, y))
    print(f"  BCE(after):  {loss_after:.6f}")
    assert loss_after < loss_before, f"{opt_name}: loss did not decrease"

    y_pred = to_numpy(model.predict(x)).reshape(-1)
    print("  preds:", np.round(y_pred, 3))


def test_optimizer_on_regression(opt_name, lr=0.005, epochs=600):
    print(f"\n=== Linear Regression y=3x-1 / {opt_name} ===")
    np.random.seed(0)

    x = np.linspace(-2, 2, 64, dtype=np.float32).reshape(64,1,1,1)
    y = (3.0 * x.reshape(64,1) - 1.0).astype(np.float32)
    y += np.random.randn(*y.shape).astype(np.float32) * 0.05

    model = build_reg_model(lr=lr, optimizer=opt_name)
    debug_graph_bindings(model)

    # 🔎 W 부호 체크: 학습 전, 소배치로 1회만
    numeric_vs_backprop_grad_W(model, x[:16], y[:16], eps=1e-3)

    loss_before = float(model.evaluate(x, y))
    print(f"  MSE(before): {loss_before:.6f}")
    dump_param_stats(model, tag="init")

    params_before = flatten_params(get_params(model))
    model.fit(x, y, epochs=1, batch_size=16)
    params_after_1 = flatten_params(get_params(model))
    delta_1 = params_after_1 - params_before

    try:
        grads = getattr(model, "last_grads", None)
        if grads is not None:
            grads_flat = flatten_params(grads)
            grad_inner = dot(delta_1, grads_flat)
            print(f"  ΔW·grad(after 1 step): {grad_inner:.6e} (expected < 0)")
    except Exception:
        pass

    remain = max(0, epochs-1)
    for e in range(remain):
        model.fit(x, y, epochs=1, batch_size=16)
        if (e+1) % 50 == 0:
            mid = float(model.evaluate(x, y))
            print(f"  [epoch+{e+1}] MSE={mid:.6f}")
            if not np.isfinite(mid):
                raise RuntimeError(f"MSE became non-finite at epoch {e+1}")
            dump_param_stats(model, tag=f"epoch+{e+1}")

    loss_after = float(model.evaluate(x, y))
    print(f"  MSE(after):  {loss_after:.6f}")
    assert loss_after < loss_before, f"{opt_name}: loss did not decrease"

    # 최종 (W, b) 출력: 일반화된 첫 W/b 키를 사용
    wkey = next((k for k in model.weights.keys() if k.endswith("_W")), None)
    bkey = next((k for k in model.biases.keys() if k.endswith("_b")), None)
    if wkey and bkey:
        Wv = float(to_numpy(model.weights[wkey]).reshape(-1)[0])
        bv = float(to_numpy(model.biases[bkey]).reshape(-1)[0])
        print(f"  learned (W, b) ≈ ({Wv:.3f}, {bv:.3f})  target ≈ (3, -1)")


# ========== 실행 ==========
if __name__ == "__main__":
    # 옵티마이저별 안정적인 기본 학습률
    for opt in ["sgd", "momentum", "adam"]:
        if opt == "sgd":
            lr_xor, lr_reg = 0.3, 0.0005   # 회귀는 더 보수적으로
        elif opt == "momentum":
            lr_xor, lr_reg = 0.2, 0.01
        else:  # adam
            lr_xor, lr_reg = 0.01, 0.001

        # 필요 시 XOR도 테스트
        # test_optimizer_on_xor(opt_name=opt, lr=lr_xor, epochs=1000)
        test_optimizer_on_regression(opt_name=opt, lr=lr_reg, epochs=600)

    print("\nAll optimizer wiring & smoke tests passed ✅")
