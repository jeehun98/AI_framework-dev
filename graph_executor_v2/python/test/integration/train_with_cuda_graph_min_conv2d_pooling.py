# examples/train_with_cuda_graph_min.py
# -------------------------------------------------------------
# 목적:
#  - Sequential.compile()로 fwd→loss→bwd→opt를 CUDA Graph로 캡처
#  - Eager 1 step vs Graph 1 step 비교
#  - N step 반복 학습 + "고정 검증 배치"로 수렴 체크
# 특징:
#  - dY 스케일 자동 감지 → grad_scale 자동 설정
#  - 출력층 로짓 스케일 다운(옵션)
# 옵션:
#  - --steps <int>: 그래프 학습 반복 횟수(기본 100)
#  - --seed <int>: 난수 시드(기본 2025)
#  - --lr, --wd: 옵티마이저 하이퍼파라미터
#  - --batch, --cin, --h, --w, --classes: 데이터/클래스 구성
#  - --f1, --f2: Conv 채널 수(두 번째는 0이면 생략)
#  - --pool: none|max|avg (각 Conv 뒤에 적용)
#  - --pool-k, --pool-s: 풀링 커널/스트라이드 (기본 2)
#  - --out-scale <float>: 마지막 Dense의 가중치 스케일(기본 0.1; 1.0이면 비활성)
#  - --fix-train-batch: 훈련 배치도 고정(과적합 sanity check)
# 요구:
#  - Conv2D/Pool2D/Dense가 forward_into/backward_into 지원
#  - SoftmaxCrossEntropy, AdamWOpt 존재
# -------------------------------------------------------------
import os, sys, argparse
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cupy as cp
from graph_executor_v2.layers.sequential import Sequential
from graph_executor_v2.layers.conv2d import Conv2D
from graph_executor_v2.layers.pool2d import Pool2D
from graph_executor_v2.layers.flatten import Flatten
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.losses.softmax_ce import SoftmaxCrossEntropy
from graph_executor_v2.optim.adamw import AdamWOpt
from graph_executor_v2.train.cuda_graph_trainer import CudaGraphTrainer
from graph_executor_v2.losses.utils import infer_grad_scale


# ------------------------- 모델 -------------------------
def make_model(
    N: int,
    Cin: int,
    H: int,
    W: int,
    *,
    f1: int = 16,
    f2: int = 0,              # 0이면 두 번째 Conv 생략
    pool: str = "max",        # none|max|avg
    pool_k: int = 2,
    pool_s: int = 2,
    hidden: int = 64,
    classes: int = 10,
    out_scale: float = 0.1
) -> Sequential:
    """
    간단 CNN:
      Conv2D(f1, 3x3, pad=1, ReLU)
      [Pool2D(k=pool_k, s=pool_s, mode=pool)]  # pool != 'none'이면
      [Conv2D(f2, 3x3, pad=1, ReLU)]
      [Pool2D(...)]                             # pool != 'none'이면 & f2>0
      Flatten
      Dense(hidden, ReLU)
      Dense(C)
    """
    layers = []
    # Conv1
    layers.append(
        Conv2D(out_channels=f1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
               dilation=(1, 1), groups=1, with_bias=True, activation="relu")
    )
    if pool and pool.lower() != "none":
        layers.append(Pool2D(kernel_size=(pool_k, pool_k), stride=(pool_s, pool_s),
                             padding=(0, 0), dilation=(1, 1),
                             ceil_mode=False, count_include_pad=False, mode=pool))

    Cin_after = f1
    # Conv2 (optional)
    if f2 > 0:
        layers.append(
            Conv2D(out_channels=f2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                   dilation=(1, 1), groups=1, with_bias=True, activation="relu")
        )
        if pool and pool.lower() != "none":
            layers.append(Pool2D(kernel_size=(pool_k, pool_k), stride=(pool_s, pool_s),
                                 padding=(0, 0), dilation=(1, 1),
                                 ceil_mode=False, count_include_pad=False, mode=pool))
        Cin_after = f2

    # Head
    layers += [
        Flatten(),                        # (N, Cin', H, W) → (N, Cin'*H*W)
        Dense(hidden, activation="relu",   initializer="he",     use_native_bwd=True),
        Dense(classes, activation="none",  initializer="xavier", use_native_bwd=True),
    ]

    m = Sequential(*layers, name="MiniCNN")
    m.build((N, Cin, H, W))
    m.train(True)

    # 출력층 스케일 다운 (초기 폭주 억제)
    if out_scale is not None and out_scale != 1.0:
        last = m.layers[-1]
        if hasattr(last, "W") and isinstance(last.W, cp.ndarray):
            last.W *= float(out_scale)

    return m


# ------------------------- 유틸 -------------------------
def eager_one_step(model, loss_fn, opt, X, y) -> float:
    """그래프 비교용 Eager 1 스텝."""
    logits = model(X)
    L, dY = loss_fn.forward(logits, y)
    model.zero_grad()
    model.backward(dY)
    if hasattr(opt, "step"):
        opt.step()
    return float(L)


# ------------------------- 메인 -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)

    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cin", type=int, default=3)
    ap.add_argument("--h", type=int, default=28)
    ap.add_argument("--w", type=int, default=28)
    ap.add_argument("--classes", type=int, default=10)

    ap.add_argument("--f1", type=int, default=16)
    ap.add_argument("--f2", type=int, default=0)          # 0이면 두 번째 Conv 생략
    ap.add_argument("--pool", type=str, default="max", choices=["none", "max", "avg"])
    ap.add_argument("--pool-k", type=int, default=2)
    ap.add_argument("--pool-s", type=int, default=2)

    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out-scale", type=float, default=0.1)

    ap.add_argument("--fix-train-batch", action="store_true")
    args = ap.parse_args()

    cp.random.seed(args.seed)
    N, Cin, H, W, C = args.batch, args.cin, args.h, args.w, args.classes

    # --- 데이터: 훈련 배치(기본 랜덤), 검증 배치(항상 고정) ---
    X_train = cp.random.randn(N, Cin, H, W).astype(cp.float32)
    y_train = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    if not args.fix_train_batch:
        pool_steps = max(args.steps, 1)
        X_pool = cp.random.randn(pool_steps, N, Cin, H, W).astype(cp.float32)
        y_pool = cp.random.randint(0, C, size=(pool_steps, N), dtype=cp.int32)
    else:
        X_pool = None
        y_pool = None

    X_val = cp.random.randn(N, Cin, H, W).astype(cp.float32)
    y_val = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    loss = SoftmaxCrossEntropy()

    # --- grad_scale 자동 감지 ---
    tmp_model = make_model(
        N, Cin, H, W,
        f1=args.f1, f2=args.f2,
        pool=args.pool, pool_k=args.pool_k, pool_s=args.pool_s,
        hidden=args.hidden, classes=C, out_scale=args.out_scale
    )
    if not getattr(tmp_model, "built", False):
        tmp_model.build((N, Cin, H, W))
    tmp_model.train(True)

    gs, mode = infer_grad_scale(loss, tmp_model, X_train, y_train)
    print(f"[GS] inferred grad_scale = {gs:.6f}")

    # --- 실제 모델/옵티마이저/트레이너 구성 ---
    model = make_model(
        N, Cin, H, W,
        f1=args.f1, f2=args.f2,
        pool=args.pool, pool_k=args.pool_k, pool_s=args.pool_s,
        hidden=args.hidden, classes=C, out_scale=args.out_scale
    )

    opt = AdamWOpt([], lr=float(args.lr), wd=float(args.wd))
    if hasattr(opt, "ensure_initialized"):
        opt.ensure_initialized()
    if hasattr(opt, "set_grad_scale"):
        opt.set_grad_scale(float(gs))

    trainer = CudaGraphTrainer(model, loss, opt)
    trainer.compile((N, Cin, H, W))  # 내부에서 파라미터 rebind 및 그래프 캡처

    # --- Eager 1 step vs Graph 1 step (검증 배치 기준으로 비교) ---
    model_eager = make_model(
        N, Cin, H, W,
        f1=args.f1, f2=args.f2,
        pool=args.pool, pool_k=args.pool_k, pool_s=args.pool_s,
        hidden=args.hidden, classes=C, out_scale=args.out_scale
    )
    opt_eager = AdamWOpt([], lr=float(args.lr), wd=float(args.wd))
    if hasattr(opt_eager, "ensure_initialized"): opt_eager.ensure_initialized()
    if hasattr(opt_eager, "set_grad_scale"):     opt_eager.set_grad_scale(float(gs))

    L_eager_before = eager_one_step(model_eager, loss, opt_eager, X_train, y_train)

    L0, _ = loss.forward(model(X_val), y_val)   # baseline(검증 고정)
    trainer.one_step(X_train, y_train)          # 그래프 경로 1스텝
    L1, _ = loss.forward(model(X_val), y_val)
    print(f"[CHK] eager step loss(before) = {L_eager_before:.6f}")
    print(f"[CHK] graph step loss(after)  = {L1:.6f}")
    print(f"[VAL] baseline (before train) loss = {float(L0):.6f}")

    # --- 반복 학습 (CUDA Graph) ---
    steps = int(args.steps)
    start_evt, end_evt = cp.cuda.Event(), cp.cuda.Event()
    cp.cuda.Stream.null.synchronize()
    start_evt.record()

    sample_points = [int(i * (steps - 1) / 10) for i in range(11)] if steps > 1 else [0]
    sampled_losses = []

    for t in range(steps):
        if X_pool is not None:
            Xt = X_pool[t]
            yt = y_pool[t]
        else:
            Xt = X_train
            yt = y_train

        trainer.one_step(Xt, yt)

        if t in sample_points:
            trainer.tg._stream.synchronize()
            Ls, _ = loss.forward(model(X_val), y_val)
            print(f"[TRAIN] step={t:04d} loss={float(Ls):.6f}")
            sampled_losses.append(float(Ls))

    end_evt.record()
    end_evt.synchronize()
    ms = cp.cuda.get_elapsed_time(start_evt, end_evt)
    print(f"[RUN] steps={steps}, elapsed={ms:.3f} ms, avg/step={ms/max(steps,1):.3f} ms")
    print(f"[VAL] losses ({len(sampled_losses)} pts): {', '.join(f'{v:.4f}' for v in sampled_losses)}")

    if len(sampled_losses) >= 2 and not (sampled_losses[-1] < sampled_losses[0]):
        print("[WARN] validation loss did not improve "
              f"(first={sampled_losses[0]:.4f}, last={sampled_losses[-1]:.4f})")


if __name__ == "__main__":
    main()


# python train_with_cuda_graph_min_conv2d_pooling.py --pool max --pool-k 2 --pool-s 2 --f1 16 --f2 16
