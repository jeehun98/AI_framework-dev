# python/test/integration/test_graph_minimal_check.py
# --------------------------------------------------
# 목적:
# 1) eager vs graph 1-step 손실 비교
# 2) graph 1-step 직후 Δθ·g(= (p_after - p_before)·g ) 내적 부호 검사 (음수여야 하강)
# 3) AdamWOpt.step_into가 step과 동일 경로를 타도록 몽키패치(응급)
# 4) 필요 시 --use-sgd 로 SGD로도 동일 체크 (AdamW 문제 분리)
# --------------------------------------------------

import os, sys, argparse
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cupy as cp

from graph_executor_v2.layers.sequential import Sequential
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.losses.softmax_ce import SoftmaxCrossEntropy
from graph_executor_v2.optim.adamw import AdamWOpt, collect_params_from
try:
    # 있을 때만 사용 (캡처 버퍼에서 gW/gB를 뽑아 옵티마와 rebind)
    from graph_executor_v2.optim.adamw import collect_params_from_plan
except Exception:
    collect_params_from_plan = None


# ---------------------- util ----------------------
def make_model(M=64, in_dim=128, hid=256, out_dim=11):
    m = Sequential(
        Dense(hid, activation="relu", initializer="he",     use_native_bwd=True, name="dense1"),
        Dense(out_dim, activation="none", initializer="xavier", use_native_bwd=True, name="dense2"),
        name="MLP_min",
    )
    m.build((M, in_dim))
    m.train(True)
    return m

def snapshot_params(model):
    snaps = []
    for lyr in model.layers:
        if hasattr(lyr, "W") and hasattr(lyr, "b"):
            snaps.append(("W", lyr.W.copy()))
            snaps.append(("b", lyr.b.copy()))
        else:
            snaps.append(("none", None))
    return snaps

def restore_params(model, snaps):
    si = 0
    for lyr in model.layers:
        if hasattr(lyr, "W") and hasattr(lyr, "b"):
            tag, w_before = snaps[si];   si += 1
            tag, b_before = snaps[si];   si += 1
            lyr.W[...] = w_before
            lyr.b[...] = b_before
        else:
            si += 1  # 'none'
            
def delta_dot_sign_from_plan(model, plan):
    """plan 캡처 버퍼의 gW/gB를 이용해 Δθ·g 합을 산출"""
    s = 0.0
    bi = 0
    for lyr in model.layers:
        if hasattr(lyr, "W") and hasattr(lyr, "b"):
            gW = plan["buffers"]["bwd"][bi]["gW"]
            gB = plan["buffers"]["bwd"][bi]["gB"]
            # Δθ = (after - before) 를 위해 W._before/b._before를 이용
            dW = lyr.W - lyr.W._before
            dB = lyr.b - lyr.b._before
            s += float(cp.vdot(dW.ravel(), gW.ravel()))
            s += float(cp.vdot(dB.ravel(), gB.ravel()))
        bi += 1
    return s

def attach_before_snap(model):
    for lyr in model.layers:
        if hasattr(lyr, "W"):
            lyr.W._before = lyr.W.copy()
        if hasattr(lyr, "b"):
            lyr.b._before = lyr.b.copy()

class SGDOptSimple:
    """그래프 진단용 초간단 SGD (step == step_into)"""
    def __init__(self, triplets, lr=1e-3, wd=0.0):
        self.groups = [{"p": p, "g": g, "exempt": ex} for (p, g, ex) in triplets]
        self.lr = float(lr); self.wd = float(wd)
        self.grad_scale = 1.0

    def rebind_grads(self, triplets):
        self.groups = [{"p": p, "g": g, "exempt": ex} for (p, g, ex) in triplets]

    def ensure_initialized(self): pass
    def set_grad_scale(self, s): self.grad_scale = float(s)

    def _apply(self):
        for s in self.groups:
            p, g, ex = s["p"], s["g"], s["exempt"]
            if not isinstance(g, cp.ndarray): continue
            if (self.wd != 0.0) and not ex:
                p -= self.lr * self.wd * p
            p -= self.lr * self.grad_scale * g

    def step(self): self._apply()
    def step_into(self): self._apply()


# ---------------------- tests ----------------------
def eager_one_step(m, loss_fn, opt, X, y):
    logits = m(X)
    L, dY = loss_fn.forward(logits, y)
    m.zero_grad()
    m.backward(dY)
    opt.step()
    return float(L)

def graph_one_step(m, loss_fn, opt, X, y, *, lr_positive_guard=True):
    # (응급) step_into가 step과 완전히 동일하도록 강제
    if hasattr(opt, "step_into"):
        opt.step_into = getattr(opt, "step")  # 몽키패치: 동일 로직 보장

    # 학습률 안전 가드
    if hasattr(opt, "set_lr") and lr_positive_guard:
        try:
            opt.set_lr(abs(getattr(opt, "lr", 1e-3)))
        except Exception:
            pass

    tg = m.compile((X.shape[0], X.shape[1]), loss=loss_fn, optimizer=opt, lt_bytes=(8 << 20))

    # rebind가 compile 내부에서 수행되지 않은 경우를 대비해 한번 더 시도
    # (AdamWOpt에 rebind_grads가 있고, collect_params_from_plan이 있을 때만)
    if hasattr(opt, "rebind_grads") and (collect_params_from_plan is not None):
        try:
            # compile() 내부에서 만든 plan을 꺼낼 수 없으니, 다시 만들어 쓴다
            plan = m.plan_capture((X.shape[0], X.shape[1]), loss_kind="softmax_ce", lt_bytes=(8 << 20))
            opt.rebind_grads(collect_params_from_plan(m, plan))
        except Exception:
            plan = None
    else:
        plan = None

    # step 전/후 스냅샷 + Δθ·g 진단
    attach_before_snap(m)
    tg.set_batch(X, y)
    tg.launch()
    logits = m(X)  # 최신 파라미터 기준
    L_after, _ = loss_fn.forward(logits, y)

    # Δθ·g 부호 체크 (plan이 없으면 스킵)
    dot_sign = None
    if plan is not None:
        dot_sign = delta_dot_sign_from_plan(m, plan)

    return float(L_after), dot_sign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--use-sgd", action="store_true", help="AdamW 대신 SGDOptSimple로 테스트")
    args = ap.parse_args()

    cp.random.seed(args.seed)
    M, in_dim, hid, out_dim = 64, 128, 256, 11
    X = cp.random.randn(M, in_dim).astype(cp.float32)
    y = cp.random.randint(0, out_dim, size=(M,), dtype=cp.int32)
    loss_fn = SoftmaxCrossEntropy()

    # ---------- EAGER ----------
    me = make_model(M, in_dim, hid, out_dim)
    if args.use_sgd:
        params = collect_params_from(me) or []
        if len(params) == 0:
            # 레이어에서 직접 수집
            for lyr in me.layers:
                if hasattr(lyr, "W") and hasattr(lyr, "b"):
                    params.append((lyr.W, cp.zeros_like(lyr.W), False))
                    params.append((lyr.b, cp.zeros_like(lyr.b), True))
        opt_e = SGDOptSimple(params, lr=abs(args.lr), wd=args.wd)
    else:
        params = collect_params_from(me)
        if len(params) == 0:
            for lyr in me.layers:
                if hasattr(lyr, "W") and hasattr(lyr, "b"):
                    params.append((lyr.W, cp.zeros_like(lyr.W), False))
                    params.append((lyr.b, cp.zeros_like(lyr.b), True))
        opt_e = AdamWOpt(params, lr=abs(args.lr), wd=args.wd)
        if hasattr(opt_e, "ensure_initialized"):
            opt_e.ensure_initialized()
        if hasattr(opt_e, "set_grad_scale"):
            opt_e.set_grad_scale(1.0 / M)

    L0 = eager_one_step(me, loss_fn, opt_e, X, y)  # 한 스텝 전 손실

    # ---------- GRAPH ----------
    mg = make_model(M, in_dim, hid, out_dim)
    if args.use_sgd:
        params_g = collect_params_from(mg) or []
        if len(params_g) == 0:
            for lyr in mg.layers:
                if hasattr(lyr, "W") and hasattr(lyr, "b"):
                    params_g.append((lyr.W, cp.zeros_like(lyr.W), False))
                    params_g.append((lyr.b, cp.zeros_like(lyr.b), True))
        opt_g = SGDOptSimple(params_g, lr=abs(args.lr), wd=args.wd)
    else:
        params_g = collect_params_from(mg)
        if len(params_g) == 0:
            for lyr in mg.layers:
                if hasattr(lyr, "W") and hasattr(lyr, "b"):
                    params_g.append((lyr.W, cp.zeros_like(lyr.W), False))
                    params_g.append((lyr.b, cp.zeros_like(lyr.b), True))
        opt_g = AdamWOpt(params_g, lr=abs(args.lr), wd=args.wd)
        if hasattr(opt_g, "ensure_initialized"):
            opt_g.ensure_initialized()
        if hasattr(opt_g, "set_grad_scale"):
            opt_g.set_grad_scale(1.0 / M)

    # 그래프 1-step 수행 및 Δθ·g 확인
    L1, dot_sign = graph_one_step(mg, loss_fn, opt_g, X, y)

    print(f"[MINI] eager one-step loss(before)   = {L0:.6f}")
    print(f"[MINI] graph one-step loss(after)    = {L1:.6f}")
    if dot_sign is not None:
        tag = "NEG (OK, descent)" if dot_sign < 0 else "POS (BAD, ascent)"
        print(f"[MINI] Σ(Δθ·g) after graph step    = {dot_sign:.6e}  --> {tag}")

    # 간단 어서션(그래프도 최소한 하강해야 정상)
    if not (L1 < 1e9):  # NaN/Inf 방지용 느슨한 체크
        raise AssertionError("Loss exploded or invalid.")
    # 비교 기준: eager에서의 전손실(L0) 대비 graph 후손실(L1)이 줄어드는지
    # (엄격 비교가 필요하면 여길 강화하세요)
    if not (L1 < L0):
        raise AssertionError(f"Graph step did not decrease loss: eager_before={L0:.6f}, graph_after={L1:.6f}")

if __name__ == "__main__":
    main()
