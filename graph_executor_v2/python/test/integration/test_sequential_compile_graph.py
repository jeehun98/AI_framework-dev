# python/test/integration/test_sequential_compile_graph.py
# --- add project root to sys.path (Windows/any) ---
import os, sys, argparse
THIS = os.path.abspath(os.path.dirname(__file__))                   # .../graph_executor_v2/python/test/integration
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))              # .../graph_executor_v2 (package root)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# --------------------------------------------------

import cupy as cp

from graph_executor_v2.layers.sequential import Sequential
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.losses.softmax_ce import SoftmaxCrossEntropy
from graph_executor_v2.optim.adamw import AdamWOpt, collect_params_from
# (compile() 내부에서 collect_params_from_plan을 사용하므로 이 파일에선 불필요)

# ---- 글로벌 인자 (간편 접근용) ----
ARGS = None  # main()에서 채움


# -------------------- utils --------------------
def max_err(a: cp.ndarray, b: cp.ndarray) -> float:
    return float(cp.max(cp.abs(a - b)))


def rel_err(a: cp.ndarray, b: cp.ndarray, eps: float = 1e-6) -> float:
    num = cp.max(cp.abs(a - b))
    den = cp.maximum(cp.max(cp.abs(a)), cp.max(cp.abs(b)))
    return float(num / (den + eps))


def clone_sequential(src: Sequential, use_native_bwd: bool) -> Sequential:
    dst_layers = []
    for lyr in src.layers:
        if isinstance(lyr, Dense):
            d = Dense(
                units=lyr.units,
                activation=lyr.activation,
                initializer="zeros",  # 곧 덮어씀
                use_native_bwd=use_native_bwd,
                name=(lyr.name + "_clone" if lyr.name else None),
                leaky_slope=getattr(lyr, "leaky_slope", 0.0),
            )
            dst_layers.append(d)
        else:
            raise NotImplementedError(f"clone_sequential: unsupported layer type {type(lyr)}")
    dst = Sequential(*dst_layers, name=(src.name + "_clone" if src.name else None))
    dst.build(src.input_shape)
    # 파라미터 복사
    for l_src, l_dst in zip(src.layers, dst.layers):
        assert isinstance(l_src, Dense) and isinstance(l_dst, Dense)
        l_dst.W[...] = l_src.W
        l_dst.b[...] = l_src.b
    return dst


# -------------------- original pair test --------------------
def test_pair_manual_vs_native(
    M=64, in_dim=128, hid=256, out_dim=11,
    act_hidden="relu",
    seed=0,
    atol=2e-5, rtol=2e-4, verbose=True,
):
    """
    Dense 2-레이어 MLP에서 manual BWD vs native BWD 동치성 점검.
    """
    cp.random.seed(seed)
    X = cp.random.randn(M, in_dim).astype(cp.float32)
    y = cp.random.randint(0, out_dim, size=(M,), dtype=cp.int32)
    criterion = SoftmaxCrossEntropy()

    model_manual = Sequential(
        Dense(hid, activation=act_hidden, initializer="he",       use_native_bwd=False, name="dense1"),
        Dense(out_dim, activation="none",   initializer="xavier", use_native_bwd=False, name="dense2"),
        name="MLP_manual",
    )
    model_manual.build((M, in_dim))

    model_native = clone_sequential(model_manual, use_native_bwd=True)

    model_manual.train(True); model_native.train(True)
    logits0 = model_manual(X)
    logits1 = model_native(X)

    e_y_abs = max_err(logits0, logits1)
    e_y_rel = rel_err(logits0, logits1)
    if verbose:
        print(f"[PAIR] forward OK: shape={logits0.shape}, abs={e_y_abs:.3e}, rel={e_y_rel:.3e}")
    assert logits0.shape == (M, out_dim) and logits1.shape == (M, out_dim)
    assert (e_y_abs < atol) or (e_y_rel < rtol), f"forward mismatch: abs={e_y_abs}, rel={e_y_rel}"

    loss0, dY0 = criterion.forward(logits0, y)
    loss1, dY1 = criterion.forward(logits1, y)
    if verbose:
        print(f"[PAIR] loss(manual={float(loss0):.6f}, native={float(loss1):.6f})")

    model_manual.zero_grad()
    model_native.zero_grad()
    model_manual.backward(dY0)
    model_native.backward(dY1)

    dW0_1, db0_1 = model_manual.layers[0].dW, model_manual.layers[0].db
    dW0_2, db0_2 = model_manual.layers[1].dW, model_manual.layers[1].db
    dW1_1, db1_1 = model_native.layers[0].dW, model_native.layers[0].db
    dW1_2, db1_2 = model_native.layers[1].dW, model_native.layers[1].db
    for tag, t in {"dW0_1": dW0_1, "db0_1": db0_1, "dW0_2": dW0_2, "db0_2": db0_2,
                   "dW1_1": dW1_1, "db1_1": db1_1, "dW1_2": dW1_2, "db1_2": db1_2}.items():
        assert t is not None, f"{tag} is None"

    def pair(tag, a, b):
        ea = max_err(a, b); er = rel_err(a, b)
        if verbose:
            print(f"  {tag:<6} abs={ea:.3e}, rel={er:.3e}, shape={a.shape}")
        assert (ea < atol) or (er < rtol), f"{tag} mismatch: abs={ea}, rel={er}"
        return ea, er

    if verbose:
        print("[PAIR] backward grads compare (manual vs native)")
    pair("W1", dW0_1, dW1_1)
    pair("b1", db0_1, db1_1)
    pair("W2", dW0_2, dW1_2)
    pair("b2", db0_2, db1_2)

    return {
        "loss_manual": float(loss0),
        "loss_native": float(loss1),
        "f_abs": e_y_abs, "f_rel": e_y_rel
    }


# -------------------- diagnostics helpers --------------------
def layer_param_stats(model: Sequential):
    """
    레이어를 직접 스캔해서 파라미터/그라드 노름을 수집 (collect 실패 우회).
    """
    stats = []
    total_p = 0.0
    total_g = 0.0
    for li, lyr in enumerate(model.layers):
        if hasattr(lyr, "W") and hasattr(lyr, "b"):
            W = lyr.W; b = lyr.b
            gW = getattr(lyr, "dW", None); gB = getattr(lyr, "db", None)
            pW = float(cp.linalg.norm(W)); pB = float(cp.linalg.norm(b))
            gWn = float(cp.linalg.norm(gW)) if gW is not None else None
            gBn = float(cp.linalg.norm(gB)) if gB is not None else None
            stats.append({
                "layer": li,
                "W_norm": pW, "b_norm": pB,
                "gW_norm": gWn, "gB_norm": gBn,
                "W_shape": tuple(W.shape), "b_shape": tuple(b.shape),
            })
            total_p += pW + pB
            if gWn is not None: total_g += gWn
            if gBn is not None: total_g += gBn
    return {"layers": stats, "total_p": total_p, "total_g": total_g}


def optimizer_group_stats(opt: AdamWOpt, max_print: int = 4):
    """
    옵티마 슬롯의 p/g/m/v 노름 샘플을 덤프.
    """
    G = getattr(opt, "groups", [])
    out = {"num_groups": len(G), "items": []}
    for i, slot in enumerate(G):
        p = slot["p"]; g = slot["g"]; m = slot["m"]; v = slot["v"]
        item = {
            "i": i,
            "p_norm": float(cp.linalg.norm(p)),
            "g_norm": float(cp.linalg.norm(g)) if isinstance(g, cp.ndarray) else None,
            "m_norm": float(cp.linalg.norm(m)),
            "v_norm": float(cp.linalg.norm(v)),
            "shape": tuple(p.shape),
            "exempt": bool(slot.get("exempt", slot.get("is_bias", False))),
        }
        if i < max_print:
            out["items"].append(item)
    return out


def collect_params_from_layers(model: Sequential):
    """
    레이어 기반 파라미터 수집 (fallback): (W, zeros_like, False), (b, zeros_like, True)
    """
    triplets = []
    for lyr in model.layers:
        if hasattr(lyr, "W") and hasattr(lyr, "b"):
            W = lyr.W; b = lyr.b
            gW = cp.zeros_like(W)
            gB = cp.zeros_like(b)
            triplets.append((W, gW, False))  # W: decay 대상
            triplets.append((b, gB, True))   # b: decay 제외
    return triplets


# -------------------- eager training (no graph) --------------------
def _make_adamw(params, lr, wd, M):
    """전역 ARGS를 참고해서 AdamW 인스턴스 생성(안전옵션/스케일 적용)."""
    args = ARGS
    if getattr(args, "adamw_safe", False):
        opt = AdamWOpt(params, lr=lr, wd=wd, beta1=0.8, beta2=0.99, eps=1e-5)
    else:
        opt = AdamWOpt(params, lr=lr, wd=wd)
    if hasattr(opt, "ensure_initialized"):
        opt.ensure_initialized()
    if hasattr(opt, "set_grad_scale"):
        if getattr(args, "no_grad_scale", False):
            opt.set_grad_scale(1.0)
        else:
            opt.set_grad_scale(1.0)
    return opt


def test_train_eager(
    M=64, in_dim=128, hid=256, out_dim=11,
    steps=40, seed=999, lr=1e-3, wd=1e-4, verbose=True
):
    cp.random.seed(seed)
    model = Sequential(
        Dense(hid, activation="relu", initializer="he",     use_native_bwd=True, name="dense1"),
        Dense(out_dim, activation="none", initializer="xavier", use_native_bwd=True, name="dense2"),
        name="MLP_eager",
    )
    model.build((M, in_dim)); model.train(True)
    loss = SoftmaxCrossEntropy()

    params = collect_params_from(model)
    if len(params) == 0:
        params = collect_params_from_layers(model)   # ⬅️ fallback
    opt = _make_adamw(params, lr, wd, M)

    X = cp.random.randn(M, in_dim).astype(cp.float32)
    y = cp.random.randint(0, out_dim, size=(M,), dtype=cp.int32)

    first_loss = last_loss = None
    for step in range(1, steps + 1):
        logits = model(X)
        cur_loss, dY = loss.forward(logits, y)
        model.zero_grad()
        model.backward(dY)
        opt.step()   # eager path
        if step % 10 == 0 or step == 1:
            # 최신 파라미터로 재계산
            Lchk, _ = loss.forward(model(X), y)
            first_loss = float(Lchk) if first_loss is None else first_loss
            last_loss  = float(Lchk)
            if verbose:
                print(f"[EAGER] step={step:03d} loss={float(Lchk):.6f}")
    assert last_loss is not None and first_loss is not None
    return {"loss_first": first_loss, "loss_last": last_loss, "ok": (last_loss < first_loss*0.9)}


# -------------------- minimal SGD (sign sanity check) --------------------
class SGDOpt:
    def __init__(self, params, lr=1e-3, wd=0.0):
        self.params = params  # [(p, g, exempt), ...]
        self.lr = lr
        self.wd = wd
    def step(self):
        for p, g, exempt in self.params:
            if self.wd != 0.0 and not exempt:
                p -= self.lr * (self.wd * p)
            p -= self.lr * g


def quick_sgd_descent_check(model, loss_fn, X, y, lr=1e-4):
    logits = model(X); L0, dY = loss_fn.forward(logits, y)
    model.zero_grad(); model.backward(dY)

    params = []
    for lyr in model.layers:
        if hasattr(lyr, "W") and hasattr(lyr, "b"):
            assert lyr.dW is not None and lyr.db is not None, "grads must exist"
            params.append((lyr.W, lyr.dW, False))
            params.append((lyr.b, lyr.db, True))
    sgd = SGDOpt(params, lr=lr, wd=0.0)
    sgd.step()

    L1, _ = loss_fn.forward(model(X), y)
    return float(L0), float(L1)


# -------------------- eager vs graph one-step equivalence --------------------
def dump_stats(model: Sequential, loss_fn: SoftmaxCrossEntropy, X: cp.ndarray, y: cp.ndarray):
    logits = model(X)
    L, _ = loss_fn.forward(logits, y)
    pnorm = 0.0
    for (p, _, _) in collect_params_from_layers(model):  # 안전하게 레이어 기반
        pnorm += float(cp.linalg.norm(p))
    return float(L), float(pnorm)


def one_step_eager(model: Sequential, loss_fn: SoftmaxCrossEntropy, opt: AdamWOpt, X: cp.ndarray, y: cp.ndarray):
    logits = model(X)
    L, dY = loss_fn.forward(logits, y)
    model.zero_grad()
    model.backward(dY)
    opt.step()
    return float(L)


def diag_one_step_equiv(seed=777, M=64, in_dim=128, hid=256, out_dim=11, lr=1e-3, wd=1e-4, verbose=True):
    """
    Eager 1 step vs Graph 1 step 파라미터 변화/손실 동등성 점검
    """
    cp.random.seed(seed)
    X = cp.random.randn(M, in_dim).astype(cp.float32)
    y = cp.random.randint(0, out_dim, size=(M,), dtype=cp.int32)
    loss_fn = SoftmaxCrossEntropy()

    def make_model():
        m = Sequential(
            Dense(hid, activation="relu",   initializer="he",     use_native_bwd=True, name="dense1"),
            Dense(out_dim, activation="none", initializer="xavier", use_native_bwd=True, name="dense2"),
            name="MLP_diag",
        )
        m.build((M, in_dim)); m.train(True)
        return m

    # Eager
    m0 = make_model()
    params0 = collect_params_from(m0)
    if len(params0) == 0:
        params0 = collect_params_from_layers(m0)
    opt0 = _make_adamw(params0, lr, wd, M)
    L0_before, P0_before = dump_stats(m0, loss_fn, X, y)
    L0_step = one_step_eager(m0, loss_fn, opt0, X, y)
    L0_after, P0_after = dump_stats(m0, loss_fn, X, y)

    # Graph
    m1 = make_model()
    params1 = collect_params_from(m1)
    if len(params1) == 0:
        params1 = collect_params_from_layers(m1)
    opt1 = _make_adamw(params1, lr, wd, M)
    tg = m1.compile((M, in_dim), loss=loss_fn, optimizer=opt1, lt_bytes=(8 << 20))
    tg.set_batch(X, y)
    tg.launch()  # 1 step
    L1_after, P1_after = dump_stats(m1, loss_fn, X, y)

    if verbose:
        print(f"[DIAG/EAGER]  before L={L0_before:.6f}, after L={L0_after:.6f}, step L={L0_step:.6f}, |θ| {P0_before:.3e}->{P0_after:.3e}")
        print(f"[DIAG/GRAPH]               after L={L1_after:.6f},                     |θ|       (n/a)->{P1_after:.3e}")

    return {
        "eager": {"L_before": L0_before, "L_after": L0_after, "L_step": L0_step, "P_before": P0_before, "P_after": P0_after},
        "graph": {"L_after": L1_after, "P_after": P1_after},
    }


# -------------------- compile + train (graph) --------------------
def _dot_theta_delta_with_grads(opt: AdamWOpt, p_before: dict) -> float:
    s = 0.0
    for slot in getattr(opt, "groups", []):
        p = slot["p"]; g = slot["g"]
        if isinstance(p, cp.ndarray) and isinstance(g, cp.ndarray):
            dp = p - p_before[id(p)]
            s += float(cp.vdot(dp.ravel(), g.ravel()))
    return s


def test_compile_and_train(
    M=64, in_dim=128, hid=256, out_dim=11,
    steps=40, seed=123, lr=1e-3, wd=1e-4, verbose=True,
    ref_fwd_log=False, sync_log=False, wd_zero=False
):
    """
    Sequential.compile() + AdamWOpt 기반 그래프 캡처 학습 검증.
    fwd→loss→bwd→opt 전부 GPU 그래프 안에서 실행되고,
    손실이 충분히 감소하는지 확인.
    """
    cp.random.seed(seed)

    model = Sequential(
        Dense(hid, activation="relu",   initializer="he",     use_native_bwd=True, name="dense1"),
        Dense(out_dim, activation="none", initializer="xavier", use_native_bwd=True, name="dense2"),
        name="MLP_compile",
    )
    model.build((M, in_dim))
    model.train(True)

    loss = SoftmaxCrossEntropy()
    use_wd = 0.0 if wd_zero else wd

    params = collect_params_from(model)
    if len(params) == 0:
        params = collect_params_from_layers(model)   # ⬅️ fallback
    opt = _make_adamw(params, lr, use_wd, M)

    # === 진단: 캡처 전 파라미터/옵티마 상태 ===
    lp = layer_param_stats(model)
    og = optimizer_group_stats(opt)
    if verbose:
        print(f"[DIAG] pre-capture param_total_norm={lp['total_p']:.6e}, grad_total_norm={lp['total_g']:.6e}")
        print(f"[DIAG] pre-capture opt_groups={og['num_groups']}, sample={og['items']}")

    # 그래프 컴파일 (입력/라벨 고정 버퍼 포함)
    tg = model.compile((M, in_dim), loss=loss, optimizer=opt, lt_bytes=(8 << 20))

    # 더미 데이터(고정 배치)
    X = cp.random.randn(M, in_dim).astype(cp.float32)
    y = cp.random.randint(0, out_dim, size=(M,), dtype=cp.int32)

    # 실행 + 로깅
    first_loss = None
    last_loss = None

    # Δθ·g 디버그: step=1에서만 체크
    p_before = {id(slot["p"]): slot["p"].copy()
                for slot in getattr(opt, "groups", [])
                if isinstance(slot["p"], cp.ndarray)}

    for step in range(1, steps + 1):
        tg.set_batch(X, y)   # 고정 포인터 버퍼에 내용만 복사
        tg.launch()          # fwd→loss→bwd→opt (CUDA Graph)

        if step == 1:
            sdot = _dot_theta_delta_with_grads(opt, p_before)
            if verbose:
                print(f"[DEBUG] Σ(Δθ·g) step=1 = {sdot:.6e}  --> {'NEG(OK)' if sdot < 0 else 'POS(BAD)'}")

        if (step % 10 == 0) or (step == 1):
            if sync_log:
                cp.cuda.Stream.null.synchronize()

            # 그래프가 제공하는 logits 버퍼 사용 vs 재-fwd
            logits = model(X) if ref_fwd_log else tg.logits
            cur_loss, _ = loss.forward(logits, y)
            cur_loss = float(cur_loss)
            if first_loss is None:
                first_loss = cur_loss
            last_loss = cur_loss

            # === 진단: 파라미터/옵티마 상태 찍기 ===
            lp_i = layer_param_stats(model)
            og_i = optimizer_group_stats(opt)
            if verbose:
                wd_tag = f"(wd={'0' if wd_zero else use_wd})"
                print(f"[CAP] step={step:03d} loss={cur_loss:.6f} {wd_tag}")
                print(f"      param_total_norm={lp_i['total_p']:.6e}, grad_total_norm={lp_i['total_g']:.6e}")
                if step in (1, 10):
                    print(f"      opt_groups={og_i['num_groups']} sample={og_i['items']}")

    assert last_loss is not None and first_loss is not None
    # 간단한 수렴 어서션: 초기 대비 일정 비율 이상 감소했는지 체크(여유롭게 10% 이상)
    assert last_loss < first_loss * 0.9, f"loss not decreased enough: first={first_loss}, last={last_loss}"

    return {"loss_first": first_loss, "loss_last": last_loss}


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--atol", type=float, default=2e-5)
    ap.add_argument("--rtol", type=float, default=2e-4)
    ap.add_argument("--no-pair", action="store_true", help="수동 vs 네이티브 BWD 테스트 생략")

    # 진단 옵션
    ap.add_argument("--diag-one-step", action="store_true", help="Eager 1 step vs Graph 1 step 동등성 체크")
    ap.add_argument("--ref-fwd-log", action="store_true", help="로깅 시 model(X) 재-Forward로 손실 계산")
    ap.add_argument("--sync-log", action="store_true", help="로깅 직전 스트림 동기화")
    ap.add_argument("--wd-zero", action="store_true", help="학습 시 weight decay=0 으로 강제")
    ap.add_argument("--train-eager", action="store_true", help="그래프 없이 eager 학습 검증")
    ap.add_argument("--check-sgd", action="store_true", help="SGD 1-step 부호 점검")

    # 🔐 안정화 토글
    ap.add_argument("--adamw-safe", action="store_true",
                    help="AdamW를 안정 하이퍼파라미터(beta1=0.8,beta2=0.99,eps=1e-5)로 사용")
    ap.add_argument("--no-grad-scale", action="store_true",
                    help="grad_scale(1/M) 비활성화")

    args = ap.parse_args()

    # 전역 ARGS 주입
    global ARGS
    ARGS = args

    if not args.no_pair:
        pair = test_pair_manual_vs_native(
            M=64, in_dim=128, hid=256, out_dim=11,
            seed=args.seed, atol=args.atol, rtol=args.rtol, verbose=True,
        )
        print("[INTEG] manual vs native:", pair)

    if args.diag_one_step:
        diag = diag_one_step_equiv(
            seed=args.seed + 777,
            M=64, in_dim=128, hid=256, out_dim=11,
            lr=args.lr,
            wd=(0.0 if args.wd_zero else args.wd),
            verbose=True
        )
        print("[DIAG] one-step equivalence:", diag)

    if args.check_sgd:
        cp.random.seed(args.seed + 2025)
        M, in_dim, hid, out_dim = 64, 128, 256, 11
        model = Sequential(
            Dense(hid, activation="relu", initializer="he",     use_native_bwd=True),
            Dense(out_dim, activation="none", initializer="xavier", use_native_bwd=True),
        )
        model.build((M, in_dim)); model.train(True)
        loss_fn = SoftmaxCrossEntropy()
        X = cp.random.randn(M, in_dim).astype(cp.float32)
        y = cp.random.randint(0, out_dim, size=(M,), dtype=cp.int32)
        L0, L1 = quick_sgd_descent_check(model, loss_fn, X, y, lr=1e-4)
        print(f"[SGD-CHECK] L0={L0:.6f} -> L1={L1:.6f}  ({'decrease ✅' if L1 < L0 else 'increase ❌'})")

    cap = test_compile_and_train(
        M=64, in_dim=128, hid=256, out_dim=11,
        steps=args.steps, seed=args.seed + 123,
        lr=args.lr, wd=args.wd, verbose=True,
        ref_fwd_log=args.ref_fwd_log,
        sync_log=args.sync_log,
        wd_zero=args.wd_zero,
    )
    print("[INTEG][CAP] ", cap)

    print("[INTEG] all good ✅")


if __name__ == "__main__":
    main()
