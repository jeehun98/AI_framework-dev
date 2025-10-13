# python/test/layers/test_dense_gemm.py
# --- add project root to sys.path (Windows/any) ---
import os, sys, argparse
THIS = os.path.abspath(os.path.dirname(__file__))                   # .../graph_executor_v2/python/test/layers
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))              # .../graph_executor_v2 (package root)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# --------------------------------------------------

import cupy as cp
from graph_executor_v2.layers.dense_gemm import Dense


def max_err(a: cp.ndarray, b: cp.ndarray) -> float:
    return float(cp.max(cp.abs(a - b)))


def rel_err(a: cp.ndarray, b: cp.ndarray, eps: float = 1e-6) -> float:
    num = cp.max(cp.abs(a - b))
    den = cp.maximum(cp.max(cp.abs(a)), cp.max(cp.abs(b)))
    return float(num / (den + eps))


def clone_dense(src: Dense, use_native_bwd: bool) -> Dense:
    """src의 파라미터를 복제하여 동등한 Dense를 만든다."""
    dst = Dense(
        units=src.units,
        activation=src.activation,
        initializer="zeros",  # 즉시 덮어쓸 거라 의미 없음
        use_native_bwd=use_native_bwd,
        name=(src.name + "_clone" if src.name else None),
        leaky_slope=src.leaky_slope,
    )
    # build with the same input dim
    assert src.W is not None and src.b is not None
    in_dim = int(src.W.shape[0])
    dst.build((0, in_dim))
    # copy weights/bias so both layers are identical
    dst.W[...] = src.W
    dst.b[...] = src.b
    return dst


def run_pair_test(M=32, in_dim=64, units=128, act="relu", seed=0,
                  atol=2e-5, rtol=2e-4, verbose=True):
    cp.random.seed(seed)

    # inputs & layer
    x = cp.random.randn(M, in_dim).astype(cp.float32)

    base = Dense(
        units=units,
        activation=act,
        initializer="he",
        use_native_bwd=False,
        name="dense_base"
    )
    base.build((M, in_dim))

    # 동일 파라미터를 공유하는 native 버전
    native = clone_dense(base, use_native_bwd=True)

    # forward 둘 다
    y0 = base(x)
    y1 = native(x)

    e_y_abs = max_err(y0, y1)
    e_y_rel = rel_err(y0, y1)
    if verbose:
        print(f"[Dense] forward OK: y.shape={y0.shape}, max_abs={e_y_abs:.3e}, rel={e_y_rel:.3e}")
    assert y0.shape == (M, units) and y1.shape == (M, units)
    assert e_y_abs < atol or e_y_rel < rtol, f"forward mismatch: abs={e_y_abs}, rel={e_y_rel}"

    # backward 입력(같은 gY)
    gy = cp.random.randn(*y0.shape).astype(cp.float32)
    dx0 = base.backward(gy)     # manual
    dx1 = native.backward(gy)   # native

    # 비교 항목: dx, dW, db
    assert dx0.shape == x.shape and dx1.shape == x.shape
    assert base.dW is not None and base.db is not None
    assert native.dW is not None and native.db is not None

    e_dx_abs = max_err(dx0, dx1); e_dx_rel = rel_err(dx0, dx1)
    e_dW_abs = max_err(base.dW, native.dW); e_dW_rel = rel_err(base.dW, native.dW)
    e_db_abs = max_err(base.db, native.db); e_db_rel = rel_err(base.db, native.db)

    if verbose:
        print(f"[Dense] backward OK (act={act})")
        print(f"  dx   : abs={e_dx_abs:.3e}, rel={e_dx_rel:.3e}, shape={dx0.shape}")
        print(f"  dW   : abs={e_dW_abs:.3e}, rel={e_dW_rel:.3e}, shape={base.dW.shape}")
        print(f"  db   : abs={e_db_abs:.3e}, rel={e_db_rel:.3e}, shape={base.db.shape}")

    # 허용 오차 내 일치성
    for tag, ea, er in [
        ("dx", e_dx_abs, e_dx_rel),
        ("dW", e_dW_abs, e_dW_rel),
        ("db", e_db_abs, e_db_rel),
    ]:
        assert ea < atol or er < rtol, f"{tag} mismatch: abs={ea}, rel={er}"

    return dict(
        e_y_abs=e_y_abs, e_y_rel=e_y_rel,
        e_dx_abs=e_dx_abs, e_dx_rel=e_dx_rel,
        e_dW_abs=e_dW_abs, e_dW_rel=e_dW_rel,
        e_db_abs=e_db_abs, e_db_rel=e_db_rel,
    )


def small_fd_check(seed=1, atol=2e-3, rtol=2e-2, act="relu"):
    """아주 작은 문제에서 수치미분으로 dW/db를 러프하게 점검(선택)."""
    cp.random.seed(seed)
    M, in_dim, units = 4, 3, 3
    x = cp.random.randn(M, in_dim).astype(cp.float32)

    layer = Dense(units=units, activation=act, initializer="small_uniform", use_native_bwd=True)
    layer.build((M, in_dim))
    y = layer(x)

    gy = cp.random.randn(*y.shape).astype(cp.float32)
    dx = layer.backward(gy)  # fills dW, db

    # finite diff for one element each to keep it quick
    eps = 1e-3

    # dW[i,j]
    i, j = 0, 0
    Wp = layer.W.copy(); Wm = layer.W.copy()
    Wp[i, j] += eps; Wm[i, j] -= eps
    layer.W[...] = Wp; y_p = layer(x)
    layer.W[...] = Wm; y_m = layer(x)
    layer.W[...] = Wp; layer.W[i, j] -= eps  # restore
    loss_p = float(cp.sum(y_p * gy))
    loss_m = float(cp.sum(y_m * gy))
    dW_fd = (loss_p - loss_m) / (2 * eps)
    e_w = abs(dW_fd - float(layer.dW[i, j]))
    print(f"[FD] dW[{i},{j}] abs_err={e_w:.3e}")

    # db[0,j]
    j = 0
    bp = layer.b.copy(); bm = layer.b.copy()
    bp[0, j] += eps; bm[0, j] -= eps
    layer.b[...] = bp; y_p = layer(x)
    layer.b[...] = bm; y_m = layer(x)
    layer.b[...] = bp; layer.b[0, j] -= eps  # restore
    loss_p = float(cp.sum(y_p * gy))
    loss_m = float(cp.sum(y_m * gy))
    db_fd = (loss_p - loss_m) / (2 * eps)
    e_b = abs(db_fd - float(layer.db[0, j]))
    print(f"[FD] db[0,{j}] abs_err={e_b:.3e}")

    assert e_w < atol, f"dW FD mismatch: {e_w}"
    assert e_b < atol, f"db FD mismatch: {e_b}"
    return dict(dw_abs=e_w, db_abs=e_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--act", type=str, default="relu",
                    choices=["none", "relu", "leakyrelu", "tanh", "sigmoid", "gelu"])
    ap.add_argument("--atol", type=float, default=2e-5)
    ap.add_argument("--rtol", type=float, default=2e-4)
    ap.add_argument("--small-fd", action="store_true", help="작은 문제에서 빠른 수치미분 체크")
    args = ap.parse_args()

    stats = run_pair_test(
        M=32, in_dim=64, units=128, act=args.act,
        seed=args.seed, atol=args.atol, rtol=args.rtol, verbose=True,
    )
    print("[Dense] manual vs native: ", stats)

    if args.small_fd:
        fd = small_fd_check(seed=args.seed + 1, act=args.act)
        print("[Dense][FD] ", fd)

    print("[Dense] all good ✅")


if __name__ == "__main__":
    main()
