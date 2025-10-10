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
    from graph_executor_v2.optim.adamw import collect_params_from_plan
except Exception:
    collect_params_from_plan = None

# ---------------- utils ----------------
def make_model(M=64, D=128, H=256, C=11):
    m = Sequential(
        Dense(H, activation="relu",   initializer="he",     use_native_bwd=True),
        Dense(C, activation="none",   initializer="xavier", use_native_bwd=True),
        name="MLP_min",
    )
    m.build((M, D)); m.train(True)
    return m

def ensure_params(model):
    ps = collect_params_from(model)
    if not ps:
        ps = []
        for lyr in model.layers:
            if hasattr(lyr, "W") and hasattr(lyr, "b"):
                ps.append((lyr.W, cp.zeros_like(lyr.W), False))
                ps.append((lyr.b, cp.zeros_like(lyr.b), True))
    return ps

class SGDOptSimple:
    """그래프 진단용 초간단 SGD (step == step_into)."""
    def __init__(self, triplets, lr=1e-3, wd=0.0, grad_scale=1.0):
        self.groups = [{"p": p, "g": g, "exempt": ex} for (p, g, ex) in triplets]
        self.lr = float(abs(lr)); self.wd = float(wd)
        self.grad_scale = float(grad_scale)
    def rebind_grads(self, triplets):
        self.groups = [{"p": p, "g": g, "exempt": ex} for (p, g, ex) in triplets]
    def ensure_initialized(self): pass
    def set_grad_scale(self, s): self.grad_scale = float(s)
    def _apply(self):
        for slot in self.groups:
            p, g, ex = slot["p"], slot["g"], slot["exempt"]
            if not isinstance(g, cp.ndarray): continue
            if (self.wd != 0.0) and not ex:
                p -= self.lr * self.wd * p
            p -= self.lr * self.grad_scale * g
    def step(self): self._apply()
    def step_into(self): self._apply()

def adamw_safe_monkeypatch(opt: AdamWOpt):
    """안전 AdamW: bias-correction 비활성화(폭주 방지), step==step_into."""
    def _apply():
        lr    = float(abs(getattr(opt, "lr", 1e-3)))
        beta1 = float(getattr(opt, "beta1", 0.9))
        beta2 = float(getattr(opt, "beta2", 0.999))
        eps   = float(getattr(opt, "eps", 1e-8))
        gscale= float(getattr(opt, "grad_scale", 1.0))
        for slot in getattr(opt, "groups", []):
            p = slot["p"]; g = slot["g"]; m = slot["m"]; v = slot["v"]
            exempt = bool(slot.get("exempt", slot.get("is_bias", False)))
            wd = float(getattr(opt, "wd", 0.0)) if not exempt else 0.0
            if not isinstance(g, cp.ndarray):  continue
            grad = g * gscale
            # 1) 1st/2nd moment
            m[...] = beta1*m + (1-beta1)*grad
            v[...] = beta2*v + (1-beta2)*(grad*grad)
            # 2) decoupled weight decay
            if wd != 0.0:
                p -= lr * wd * p
            # 3) bias-correction 없이 안정 업데이트
            p -= lr * (m / (cp.sqrt(v) + eps))
    opt.step = _apply
    opt.step_into = _apply
    # lr 양수 보장
    if hasattr(opt, "set_lr"):
        try: opt.set_lr(abs(getattr(opt, "lr", 1e-3)))
        except Exception: opt.lr = abs(getattr(opt, "lr", 1e-3))
    else:
        opt.lr = abs(getattr(opt, "lr", 1e-3))

def eager_one_step(m, loss_fn, opt, X, y):
    logits = m(X); L, dY = loss_fn.forward(logits, y)
    m.zero_grad(); m.backward(dY); opt.step()
    return float(L)

def graph_one_step(m, loss_fn, opt, X, y):
    # compile 전에 rebind 준비(있으면): 실제 compile에서도 다시 바인딩될 수 있음
    if hasattr(opt, "rebind_grads") and (collect_params_from_plan is not None):
        plan_tmp = m.plan_capture((X.shape[0], X.shape[1]), loss_kind="softmax_ce", lt_bytes=(8<<20))
        opt.rebind_grads(collect_params_from_plan(m, plan_tmp))
    tg = m.compile((X.shape[0], X.shape[1]), loss=loss_fn, optimizer=opt, lt_bytes=(8<<20))
    # Δθ·g 계산을 위해 p-before 저장
    p_before = {}
    for slot in getattr(opt, "groups", []):
        p = slot["p"]
        if isinstance(p, cp.ndarray):
            p_before[id(p)] = p.copy()
    tg.set_batch(X, y); tg.launch()
    L_after, _ = loss_fn.forward(m(X), y)
    sdot = 0.0
    for slot in getattr(opt, "groups", []):
        p, g = slot["p"], slot["g"]
        if not (isinstance(p, cp.ndarray) and isinstance(g, cp.ndarray)): continue
        dp = p - p_before[id(p)]
        sdot += float(cp.vdot(dp.ravel(), g.ravel()))
    return float(L_after), sdot

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--adamw-safe", action="store_true", help="AdamW(안전 패치)로 검사. 기본은 SGD.")
    args = ap.parse_args()

    cp.random.seed(args.seed)
    M,D,H,C = 64,128,256,11
    X = cp.random.randn(M, D).astype(cp.float32)
    y = cp.random.randint(0, C, size=(M,), dtype=cp.int32)
    loss = SoftmaxCrossEntropy()

    # ---------- EAGER ----------
    me = make_model(M,D,H,C)
    if args.adamw_safe:
        opt_e = AdamWOpt(ensure_params(me), lr=abs(args.lr), wd=args.wd)
        if hasattr(opt_e, "ensure_initialized"): opt_e.ensure_initialized()
        if hasattr(opt_e, "set_grad_scale"):     opt_e.set_grad_scale(1.0/M)
        # eager는 원래 구현 그대로 사용 (문제 분리)
    else:
        opt_e = SGDOptSimple(ensure_params(me), lr=abs(args.lr), wd=args.wd, grad_scale=1.0/M)
    L0 = eager_one_step(me, loss, opt_e, X, y)

    # ---------- GRAPH ----------
    mg = make_model(M,D,H,C)
    if args.adamw_safe:
        opt_g = AdamWOpt(ensure_params(mg), lr=abs(args.lr), wd=args.wd)
        if hasattr(opt_g, "ensure_initialized"): opt_g.ensure_initialized()
        if hasattr(opt_g, "set_grad_scale"):     opt_g.set_grad_scale(1.0/M)
        # 안전 패치(무한 재귀/폭주 방지)
        adamw_safe_monkeypatch(opt_g)
    else:
        opt_g = SGDOptSimple(ensure_params(mg), lr=abs(args.lr), wd=args.wd, grad_scale=1.0/M)

    L1, sdot = graph_one_step(mg, loss, opt_g, X, y)

    print(f"[SAFE2] eager step loss(before) = {L0:.6f}")
    print(f"[SAFE2] graph step loss(after)  = {L1:.6f}")
    print(f"[SAFE2] Σ(Δθ·g) after graph     = {sdot:.6e}  --> {'NEG(OK)' if sdot < 0 else 'POS(BAD)'}")

    # 최소 기준: 그래프 후 손실이 eager 이전 손실보다 작아야 정상
    if not (L1 < L0):
        raise AssertionError(f"Graph step did not decrease loss: eager_before={L0:.6f}, graph_after={L1:.6f}")

if __name__ == "__main__":
    main()
