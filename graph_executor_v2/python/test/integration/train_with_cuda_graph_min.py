# examples/train_with_cuda_graph_min.py
# -------------------------------------------------------------
# 목적:
#  - Sequential.compile()로 fwd→loss→bwd→opt를 CUDA Graph로 캡처
#  - Eager 1 step vs Graph 1 step 비교
#  - N step 반복 학습 + "고정 검증 배치"로 수렴 체크
# 특징:
#  - dY 스케일(sum/mean) 자동 감지 → grad_scale 자동 설정
#  - 전역 grad norm 클리핑(옵션)
#  - 출력층 로짓 스케일 다운(옵션)
# 옵션:
#  - --fix-train-batch: 훈련 배치도 고정(과적합 sanity check)
#  - --out-scale <float>: 출력층 가중치 스케일 다운(기본 0.1)
#  - --clip <float>: 전역 grad max-norm 클리핑(기본 0=off)
# 요구:
#  - Dense가 forward_into/backward_into 지원
#  - SoftmaxCrossEntropy, AdamWOpt 존재(미존재 시 간단 SGD 폴백)
# -------------------------------------------------------------
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

# ---------------- 최소 유틸 ----------------
def make_model(M=64, D=128, H=256, C=11):
    m = Sequential(
        Dense(H, activation="relu",   initializer="he",     use_native_bwd=True),
        Dense(C, activation="none",   initializer="xavier", use_native_bwd=True),
        name="MLP_min",
    )
    m.build((M, D))
    m.train(True)
    return m

def ensure_params(model):
    ps = collect_params_from(model)
    if ps:
        return ps
    out = []
    for lyr in model.layers:
        if hasattr(lyr, "W") and hasattr(lyr, "b"):
            out.append((lyr.W, cp.zeros_like(lyr.W), False))
            out.append((lyr.b, cp.zeros_like(lyr.b), True))
    return out

def infer_grad_scale(loss_fn, model, X, y):
    """
    dY 스케일(sum/mean)을 감지해 권장 grad_scale을 반환.
    - sum 스케일: ||dY||_1 ~ O(B) → 1/B 권장
    - mean 스케일: ||dY||_1 ~ O(1) → 1.0 권장
    """
    logits = model(X)
    _, dY = loss_fn.forward(logits, y)
    B = X.shape[0]
    s_mean = float(cp.abs(dY).mean())
    s_sum  = float(cp.abs(dY).sum())
    # 느슨한 기준: 합이 B에 비례하면 sum 스케일로 판단
    if s_sum > 10.0 and (s_sum / max(1e-6, B)) > 0.05:
        return 1.0 / B, {"scale":"sum", "mean":s_mean, "sum":s_sum}
    else:
        return 1.0, {"scale":"mean", "mean":s_mean, "sum":s_sum}

class SGDOptSimple:
    """그래프 검증용 초간단 SGD (decoupled WD, grad_scale, 전역 클리핑 지원)."""
    def __init__(self, triplets, lr=1e-3, wd=0.0, grad_scale=1.0, clip_max_norm=0.0):
        self.groups = [{"p": p, "g": g, "exempt": ex} for (p, g, ex) in triplets]
        self.lr = float(abs(lr)); self.wd = float(wd)
        self.grad_scale = float(grad_scale)
        self.clip_max_norm = float(max(0.0, clip_max_norm))
    def rebind_grads(self, triplets):
        self.groups = [{"p": p, "g": g, "exempt": ex} for (p, g, ex) in triplets]
    def ensure_initialized(self): pass
    def set_grad_scale(self, s: float): self.grad_scale = float(s)
    def set_clip(self, c: float): self.clip_max_norm = float(max(0.0, c))
    def _apply(self):
        lr, wd, gs, cmn = self.lr, self.wd, self.grad_scale, self.clip_max_norm
        # 전역 grad 노름
        clip_coef = 1.0
        if cmn > 0.0:
            g2 = 0.0
            for slot in self.groups:
                g = slot["g"]
                if isinstance(g, cp.ndarray):
                    g2 += float(cp.vdot(g.ravel(), g.ravel()))
            gnorm = (g2 ** 0.5) * gs
            if gnorm > cmn:
                clip_coef = cmn / (gnorm + 1e-12)

        for slot in self.groups:
            p, g, ex = slot["p"], slot["g"], slot["exempt"]
            if not isinstance(g, cp.ndarray): continue
            grad = g * gs * clip_coef
            if wd != 0.0 and not ex:
                p -= lr * wd * p
            p -= lr * grad
    def step(self): self._apply()
    def step_into(self): self._apply()

def adamw_safe_monkeypatch(opt: AdamWOpt):
    """AdamW 안전 패치: bias-correction 제거, step==step_into, 전역 클리핑 지원."""
    def _apply():
        lr    = float(abs(getattr(opt, "lr", 1e-3)))
        beta1 = float(getattr(opt, "beta1", 0.9))
        beta2 = float(getattr(opt, "beta2", 0.999))
        eps   = float(getattr(opt, "eps", 1e-8))
        gscale= float(getattr(opt, "grad_scale", 1.0))
        wd    = float(getattr(opt, "wd", 0.0))
        cmn   = float(getattr(opt, "clip_max_norm", 0.0))

        # 전역 grad 노름
        clip_coef = 1.0
        if cmn > 0.0:
            g2 = 0.0
            for slot in getattr(opt, "groups", []):
                g = slot["g"]
                if isinstance(g, cp.ndarray):
                    g2 += float(cp.vdot(g.ravel(), g.ravel()))
            gnorm = (g2 ** 0.5) * gscale
            if gnorm > cmn:
                clip_coef = cmn / (gnorm + 1e-12)

        for slot in getattr(opt, "groups", []):
            p = slot["p"]; g = slot["g"]; m = slot["m"]; v = slot["v"]
            exempt = bool(slot.get("exempt", slot.get("is_bias", False)))
            if not isinstance(g, cp.ndarray):
                continue
            grad = g * gscale * clip_coef
            m[...] = beta1*m + (1-beta1)*grad
            v[...] = beta2*v + (1-beta2)*(grad*grad)
            if (wd != 0.0) and (not exempt):
                p -= lr * wd * p
            p -= lr * (m / (cp.sqrt(v) + eps))
    opt.step = _apply
    opt.step_into = _apply
    if hasattr(opt, "set_lr"):
        try: opt.set_lr(abs(getattr(opt, "lr", 1e-3)))
        except Exception: opt.lr = abs(getattr(opt, "lr", 1e-3))
    else:
        opt.lr = abs(getattr(opt, "lr", 1e-3))
    setattr(opt, "set_clip", lambda c: setattr(opt, "clip_max_norm", float(max(0.0, c))))
    if not hasattr(opt, "clip_max_norm"):
        opt.clip_max_norm = 0.0

# ---------------- 단일 step/컴파일 ----------------
def eager_one_step(m, loss_fn, opt, X, y):
    logits = m(X)
    L, dY = loss_fn.forward(logits, y)
    m.zero_grad()
    m.backward(dY)
    opt.step()
    return float(L)

def compile_graph(model, loss_fn, optimizer, X):
    """Sequential.compile() 경로를 분리해 재사용."""
    if hasattr(optimizer, "rebind_grads") and (collect_params_from_plan is not None):
        plan_tmp = model.plan_capture((X.shape[0], X.shape[1]), loss_kind="softmax_ce", lt_bytes=(8<<20))
        optimizer.rebind_grads(collect_params_from_plan(model, plan_tmp))
    tg = model.compile((X.shape[0], X.shape[1]), loss=loss_fn, optimizer=optimizer, lt_bytes=(8<<20))
    return tg

def graph_step_once(tg, model, loss_fn, optimizer, X, y):
    # 파라미터 변화량·내적 확인용 백업
    p_before = {}
    for slot in getattr(optimizer, "groups", []):
        p = slot["p"]
        if isinstance(p, cp.ndarray):
            p_before[id(p)] = p.copy()

    # 동일 스트림에서 배치 복사 + 그래프 실행
    with tg._stream:
        tg.set_batch(X, y)
        tg.launch()
    tg._stream.synchronize()

    L_after, _ = loss_fn.forward(model(tg.X_buf), tg.y_buf)
    sdot = 0.0
    for slot in getattr(optimizer, "groups", []):
        p, g = slot["p"], slot["g"]
        if not (isinstance(p, cp.ndarray) and isinstance(g, cp.ndarray)):
            continue
        dp = p - p_before[id(p)]
        sdot += float(cp.vdot(dp.ravel(), g.ravel()))
    return float(L_after), sdot

# ---------------- 모니터링 ----------------
def norms_snapshot(optimizer):
    """||grad||, ||Δθ|| 계산(Δθ는 이전 스냅샷과의 차)."""
    g2 = 0.0; dp2 = 0.0
    for slot in getattr(optimizer, "groups", []):
        p, g = slot.get("p"), slot.get("g")
        if isinstance(g, cp.ndarray):
            g2 += float(cp.vdot(g.ravel(), g.ravel()))
        if isinstance(p, cp.ndarray) and ("_p_prev" in slot):
            dp = p - slot["_p_prev"]
            dp2 += float(cp.vdot(dp.ravel(), dp.ravel()))
        if isinstance(p, cp.ndarray):
            slot["_p_prev"] = p.copy()
    return g2 ** 0.5, dp2 ** 0.5

# ---------------- 메인 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--adamw-safe", action="store_true",
                    help="AdamW(안전 패치) 사용. 미지정 시 간단 SGD 사용")
    ap.add_argument("--steps", type=int, default=50, help="CUDA Graph 반복 스텝 수")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--din", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--classes", type=int, default=11)
    ap.add_argument("--fix-train-batch", action="store_true",
                    help="훈련 배치도 고정(과적합 sanity check)")
    ap.add_argument("--dbg-every", type=int, default=10, help="노름 출력 주기")
    ap.add_argument("--out-scale", type=float, default=0.1,
                    help="출력층 W 스케일(0.1 권장, 1.0이면 원본)")
    ap.add_argument("--clip", type=float, default=0.0,
                    help="전역 grad clip max-norm (0=off)")
    args = ap.parse_args()

    cp.random.seed(args.seed)
    M, D, H, C = args.batch, args.din, args.hidden, args.classes

    # 데이터
    X = cp.random.randn(M, D).astype(cp.float32)
    y = cp.random.randint(0, C, size=(M,), dtype=cp.int32)
    loss = SoftmaxCrossEntropy()

    # --- dY 스케일 감지 → grad_scale 추천 ---
    probe_model = make_model(M, D, H, C)
    gs_reco, gs_stats = infer_grad_scale(loss, probe_model, X, y)
    print(f"[GS] inferred grad_scale = {gs_reco:.6f}  (dY_scale={gs_stats['scale']}, "
          f"mean|dY|={gs_stats['mean']:.4e}, sum|dY|={gs_stats['sum']:.4e})")

    # ---------- EAGER 1 step ----------
    me = make_model(M, D, H, C)
    if args.adamw_safe:
        opt_e = AdamWOpt(ensure_params(me), lr=abs(args.lr), wd=args.wd)
        if hasattr(opt_e, "ensure_initialized"): opt_e.ensure_initialized()
        if hasattr(opt_e, "set_grad_scale"):     opt_e.set_grad_scale(gs_reco)
        if hasattr(opt_e, "set_clip"):           opt_e.set_clip(args.clip)
    else:
        opt_e = SGDOptSimple(ensure_params(me), lr=abs(args.lr), wd=args.wd,
                             grad_scale=gs_reco, clip_max_norm=args.clip)
    L_eager_before = eager_one_step(me, loss, opt_e, X, y)

    # ---------- GRAPH compile & 1 step ----------
    mg = make_model(M, D, H, C)

    # 출력층 로짓 스케일 다운(과신 완화)
    try:
        if args.out_scale != 1.0 and hasattr(mg.layers[-1], "W"):
            mg.layers[-1].W *= float(args.out_scale)
    except Exception:
        pass

    if args.adamw_safe:
        opt_g = AdamWOpt(ensure_params(mg), lr=abs(args.lr), wd=args.wd)
        if hasattr(opt_g, "ensure_initialized"): opt_g.ensure_initialized()
        if hasattr(opt_g, "set_grad_scale"):     opt_g.set_grad_scale(gs_reco)
        adamw_safe_monkeypatch(opt_g)  # 안전 패치 + 전역 클리핑 지원
        if hasattr(opt_g, "set_clip"):           opt_g.set_clip(args.clip)
    else:
        opt_g = SGDOptSimple(ensure_params(mg), lr=abs(args.lr), wd=args.wd,
                             grad_scale=gs_reco, clip_max_norm=args.clip)

    assert mg.supports_capture(), "모든 레이어가 forward_into/backward_into 를 구현해야 합니다."

    tg = compile_graph(mg, loss, opt_g, X)
    L_graph_after, sdot = graph_step_once(tg, mg, loss, opt_g, X, y)
    print(f"[CHK] eager step loss(before) = {L_eager_before:.6f}")
    print(f"[CHK] graph step loss(after)  = {L_graph_after:.6f}")
    print(f"[CHK] Σ(Δθ·g) after graph     = {sdot:.6e}  --> {'NEG(OK)' if sdot < 0 else 'POS(BAD)'}")
    if not (L_graph_after < L_eager_before):
        raise AssertionError(
            f"Graph step did not decrease loss: eager_before={L_eager_before:.6f}, graph_after={L_graph_after:.6f}"
        )

    # ---------- 고정 검증 배치 ----------
    X_val = cp.random.randn(M, D).astype(cp.float32)
    y_val = cp.random.randint(0, C, size=(M,), dtype=cp.int32)
    tg._stream.synchronize()
    L_val0, _ = loss.forward(mg(X_val), y_val)
    print(f"[VAL] baseline (before train) loss = {float(L_val0):.6f}")

    # ---------- GRAPH 반복 학습 ----------
    steps = int(args.steps)
    if args.fix_train_batch:
        X_pool = None; y_pool = None
    else:
        X_pool = cp.random.randn(steps, M, D).astype(cp.float32)
        y_pool = cp.random.randint(0, C, size=(steps, M), dtype=cp.int32)

    # 노름 초기 스냅샷
    for s in getattr(opt_g, "groups", []):
        if isinstance(s.get("p", None), cp.ndarray):
            s["_p_prev"] = s["p"].copy()

    start_evt, end_evt = cp.cuda.Event(), cp.cuda.Event()
    cp.cuda.Stream.null.synchronize()
    start_evt.record()

    val_losses = []
    for t in range(steps):
        with tg._stream:
            if args.fix_train_batch:
                tg.set_batch(X, y)                    # 고정 훈련 배치
            else:
                tg.set_batch(X_pool[t], y_pool[t])    # 다양한 배치
            tg.launch()

        # 모니터링(주기)
        if args.dbg_every > 0 and (t % args.dbg_every == 0):
            tg._stream.synchronize()
            gn, dn = norms_snapshot(opt_g)
            print(f"[DBG] step={t:04d} ||grad||={gn:.3e} ||Δθ||={dn:.3e}")

        # 고정 검증 배치 평가(주기)
        if (t % max(1, steps // 10) == 0) or (t == steps-1):
            tg._stream.synchronize()
            L_val, _ = loss.forward(mg(X_val), y_val)
            val_losses.append(float(L_val))

        # 고정 훈련 배치인 경우, 주기적으로 train loss 출력
        if args.fix_train_batch and (t % 10 == 0):
            tg._stream.synchronize()
            L_tr, _ = loss.forward(mg(X), y)
            print(f"[TRAIN] step={t:04d} loss={float(L_tr):.6f}")

    end_evt.record(); end_evt.synchronize()
    ms = cp.cuda.get_elapsed_time(start_evt, end_evt)

    print(f"[RUN] steps={steps}, elapsed={ms:.3f} ms, avg/step={ms/steps:.3f} ms")
    print(f"[VAL] losses ({len(val_losses)} pts): {', '.join(f'{v:.4f}' for v in val_losses)}")

    if len(val_losses) >= 1 and not (val_losses[-1] < float(L_val0)):
        print("[WARN] validation loss did not improve "
              f"(baseline={float(L_val0):.4f}, last={val_losses[-1]:.4f})")

if __name__ == "__main__":
    main()
