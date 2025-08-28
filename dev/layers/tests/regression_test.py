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
# graph_executor .pyd 경로 (프로젝트 구조에 맞춰 조정)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend", "graph_executor", "build", "lib.win-amd64-cpython-312"))
# 프로젝트 루트
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev"))

# ===== 프레임워크 임포트 =====
from dev.models.sequential import Sequential
from dev.layers.dense import Dense
from dev.layers.flatten import Flatten
import graph_executor as ge  # wiring/grad 점검용


# ---------- 유틸 ----------
def to_numpy(x):
    if isinstance(x, np.ndarray): return x
    try:
        import cupy as cp  # noqa
        if isinstance(x, cp.ndarray): return cp.asnumpy(x)
    except Exception:
        pass
    if hasattr(x, "get"):
        try: return x.get()
        except Exception: pass
    if hasattr(x, "cpu"):
        try: return x.cpu().numpy()
        except Exception: pass
    return np.asarray(x)

def flatten_params(params_dict):
    flats = []
    for k in sorted(params_dict.keys()):
        arr_np = to_numpy(params_dict[k]).astype(np.float32, copy=False)
        flats.append(arr_np.ravel())
    if flats:
        return np.concatenate(flats)
    return np.zeros(0, dtype=np.float32)

def dot(a, b):
    a = to_numpy(a).astype(np.float32, copy=False).ravel()
    b = to_numpy(b).astype(np.float32, copy=False).ravel()
    if a.size == 0 or b.size == 0: return 0.0
    return float(np.dot(a, b))

def get_params(model):
    params = {}
    params.update(getattr(model, "weights", {}))
    params.update(getattr(model, "biases", {}))
    return params

def dump_param_stats(model, tag=""):
    params = get_params(model)
    stats = {}
    for k, v in params.items():
        v_np = to_numpy(v)
        if v_np.size == 0:
            stats[k] = (0.0, 0.0, 0.0)
        else:
            stats[k] = (float(np.linalg.norm(v_np)), float(v_np.min()), float(v_np.max()))
    print(f"[{tag}] param stats (||·||2, min, max): {stats}")

def debug_graph_bindings(model):
    uses_param = {ge.OpType.MATMUL, ge.OpType.ADD, ge.OpType.CONV2D}
    graph_param_ids = sorted({op.param_id for op in model.E if (op.op_type in uses_param) and getattr(op, "param_id", "")})
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
    if miss_tensors: print("⚠️  MISSING in tensors:", miss_tensors)
    if miss_shapes:  print("⚠️  MISSING in shapes:", miss_shapes)
    if not graph_param_ids:
        print("❌ Graph has NO trainable params (no MATMUL/ADD/CONV2D with param_id).")
    elif not tensor_keys:
        print("❌ Model has NO param buffers in weights/biases.")
    elif not miss_tensors and not miss_shapes:
        print("✅ Wiring looks good.")

def quick_grad_sign_check(model, x, y, eps=1e-3):
    wkey = sorted(model.weights.keys())[0]
    bkey = sorted(model.biases.keys())[0]

    def numeric_grad_on(key, idx=0):
        arr = model.weights[key] if key == wkey else model.biases[key]
        orig = float(cp.asnumpy(arr).ravel()[idx])
        arr.ravel()[idx] = orig + eps
        loss_plus = float(model.evaluate(x, y))
        arr.ravel()[idx] = orig - eps
        loss_minus = float(model.evaluate(x, y))
        arr.ravel()[idx] = orig
        return (loss_plus - loss_minus) / (2*eps)

    x_cp = cp.asarray(x, dtype=cp.float32)
    y_cp = cp.asarray(y, dtype=cp.float32)

    tensor_ptrs = {"input": x_cp.data.ptr, "y_true": y_cp.data.ptr}
    for name, arr in model.weights.items(): tensor_ptrs[name] = arr.data.ptr
    for name, arr in model.biases.items():  tensor_ptrs[name] = arr.data.ptr

    # (선택) 순전파 1회 — 동일 dict 사용
    out_shape = model.shapes[model.output_var]
    out_host = np.zeros((x_cp.shape[0], int(out_shape.rows * out_shape.cols)), dtype=np.float32)
    ge.run_graph_forward_entry(
        E=model.E, tensors=tensor_ptrs, shapes=model.shapes,
        out_host=out_host, final_output_id=model.output_var, batch_size=x_cp.shape[0]
    )

    # ✅ 역전파: 리턴값을 받아 사용
    grads_in = {}
    ret = ge.run_graph_backward_entry(
        E=model.E, tensors=tensor_ptrs, shapes=model.shapes,
        gradients=grads_in, final_output_id=model.output_var, batch_size=x_cp.shape[0]
    )
    grads_dict = ret or grads_in   # 리턴 우선, fallback to in-place

    def pick_grad_numpy(key, shape):
        if key not in grads_dict:
            raise KeyError(f"Grad for '{key}' not found")
        ptr = int(grads_dict[key])
        nbytes = shape.rows * shape.cols * 4
        mem = cp.cuda.UnownedMemory(ptr, nbytes, model)
        mp  = cp.cuda.MemoryPointer(mem, 0)
        gcp = cp.ndarray((shape.rows, shape.cols), dtype=cp.float32, memptr=mp)
        return cp.asnumpy(gcp)

    gn_w = numeric_grad_on(wkey, 0)
    gn_b = numeric_grad_on(bkey, 0)
    gw = pick_grad_numpy(wkey, model.shapes[wkey]).ravel()[0]
    gb = pick_grad_numpy(bkey, model.shapes[bkey]).ravel()[0]
    print(f"[GradCheck] W: numeric={gn_w:.6e} backprop={gw:.6e} same? {np.sign(gn_w)==np.sign(gw)}")
    print(f"[GradCheck] b: numeric={gn_b:.6e} backprop={gb:.6e} same? {np.sign(gn_b)==np.sign(gb)}")

# ---------- 모델 ----------
def build_reg_model(lr, optimizer):
    # 완전 선형: Flatten -> Dense(1)
    model = Sequential(input_shape=(1, 1, 1))
    model.add(Flatten(input_shape=(1, 1, 1)))
    model.add(Dense(units=1, activation=None, initializer="xavier"))
    model.compile(optimizer=optimizer, loss="mse", learning_rate=lr)
    return model


# ---------- 테스트 본문 ----------
def test_regression_only(opt_name="sgd", lr=0.05, epochs=300, batch=16, do_gradcheck=True):
    """
    선형 회귀 y=3x-1, 완전 선형 모델(Flatten->Dense(1)).
    권장 학습률: SGD=0.05~0.1, Momentum(0.9)=0.02~0.05, Adam=0.01~0.02
    """
    print(f"\n=== Linear Regression y=3x-1 / {opt_name} (lr={lr}) ===")
    np.random.seed(0)

    # 데이터: x in [-1,1], y = 3x - 1 + noise
    B = 128
    x1 = np.linspace(-1, 1, B, dtype=np.float32)
    x  = x1.reshape(B,1,1,1)
    y  = (3.0 * x1 - 1.0).astype(np.float32).reshape(B,1)
    y += np.random.randn(*y.shape).astype(np.float32) * 0.05

    model = build_reg_model(lr=lr, optimizer=opt_name)
    debug_graph_bindings(model)

    # (옵션) 빠른 그라드 부호 점검
    if do_gradcheck:
        quick_grad_sign_check(model, x, y, eps=1e-3)

    loss_before = float(model.evaluate(x, y))
    print(f"  MSE(before): {loss_before:.6f}")
    dump_param_stats(model, tag="init")

    # SGD일 때 1-step 부호 점검 (ΔW·grad < 0 기대)
    if opt_name.lower() == "sgd":
        params_before = flatten_params(get_params(model))
        model.fit(x, y, epochs=1, batch_size=batch)
        params_after  = flatten_params(get_params(model))
        delta_1 = params_after - params_before

        # 동일 배치로 backprop grad 한 번 얻어서 부호 점검
        x_b = x[:batch]
        y_b = y[:batch]
        x_cp = cp.asarray(x_b, dtype=cp.float32)
        y_cp = cp.asarray(y_b, dtype=cp.float32)
        tensor_ptrs = {"input": x_cp.data.ptr, "y_true": y_cp.data.ptr}
        for name, arr in model.weights.items(): tensor_ptrs[name] = arr.data.ptr
        for name, arr in model.biases.items():  tensor_ptrs[name] = arr.data.ptr

        grads_ptrs = {}
        ge.run_graph_backward_entry(
            E=model.E, tensors=tensor_ptrs, shapes=model.shapes,
            gradients=grads_ptrs, final_output_id=model.output_var, batch_size=x_b.shape[0]
        )
        grads_dict = grads_ptrs  # ← in-place 로 채워진 걸 사용

        grads = {}
        for k, shp in model.shapes.items():
            if k in grads_dict:
                ptr = int(grads_dict[k])
                mem = cp.cuda.UnownedMemory(ptr, shp.rows*shp.cols*4, model)
                mp  = cp.cuda.MemoryPointer(mem, 0)
                grads[k] = cp.asnumpy(cp.ndarray((shp.rows, shp.cols), dtype=cp.float32, memptr=mp))
        grad_flat = flatten_params(grads)
        print(f"  ΔW·grad(after 1 step): {dot(delta_1, grad_flat):.6e} (expected < 0)")

        # 나머지 에폭은 epochs-1 만큼 더
        remain = max(0, epochs-1)
        for e in range(remain):
            model.fit(x, y, epochs=1, batch_size=batch)
            if (e+1) % 25 == 0:
                mid = float(model.evaluate(x, y))
                print(f"  [epoch+{e+1}] MSE={mid:.6f}")
                if not np.isfinite(mid): raise RuntimeError(f"MSE became non-finite at epoch {e+1}")
    else:
        # Momentum/Adam
        for e in range(epochs):
            model.fit(x, y, epochs=1, batch_size=batch)
            if (e+1) % 25 == 0:
                mid = float(model.evaluate(x, y))
                print(f"  [epoch+{e+1}] MSE={mid:.6f}")
                if not np.isfinite(mid): raise RuntimeError(f"MSE became non-finite at epoch {e+1}")

    loss_after = float(model.evaluate(x, y))
    print(f"  MSE(after):  {loss_after:.6f}")

    # 학습된 (W,b) 추출
    wkey = sorted(model.weights.keys())[0]
    bkey = sorted(model.biases.keys())[0]
    W = float(to_numpy(model.weights[wkey]).reshape(-1)[0])
    b = float(to_numpy(model.biases[bkey]).reshape(-1)[0])
    print(f"  learned (W, b) ≈ ({W:.3f}, {b:.3f})  target ≈ (3.000, -1.000)")

    assert loss_after < loss_before, f"{opt_name}: loss did not decrease"
    return loss_before, loss_after, (W, b)


if __name__ == "__main__":
    # 권장 기본값
    cfgs = [
        ("sgd",      0.05),
        ("momentum", 0.03),
        ("adam",     0.01),
    ]
    for opt, lr in cfgs:
        try:
            test_regression_only(opt_name=opt, lr=lr, epochs=300, batch=16, do_gradcheck=True)
        except AssertionError as e:
            print("ASSERT:", e)
        except Exception as e:
            print("ERROR:", e)

    print("\nRegression-only test done ✅")
