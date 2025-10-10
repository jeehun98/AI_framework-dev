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

# python/test/integration/train_with_cuda_graph_min.py
import cupy as cp
from graph_executor_v2.layers.sequential import Sequential
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.losses.softmax_ce import SoftmaxCrossEntropy
from graph_executor_v2.optim.adamw import AdamWOpt
from graph_executor_v2.train.cuda_graph_trainer import CudaGraphTrainer
from graph_executor_v2.losses.utils import infer_grad_scale

def make_model(M=64, D=128, H=256, C=11):
    m = Sequential(
        Dense(H, activation="relu",   initializer="he",     use_native_bwd=True),
        Dense(C, activation="none",   initializer="xavier", use_native_bwd=True),
    )
    m.build((M, D)); 
    m.train(True)
    return m

def main():
    M,D,H,C = 64,128,256,11
    cp.random.seed(2025)
    X = cp.random.randn(M, D).astype(cp.float32)
    y = cp.random.randint(0, C, size=(M,), dtype=cp.int32)

    loss = SoftmaxCrossEntropy()
    # grad_scale 자동 감지 (선택)
    gs, _ = infer_grad_scale(loss, make_model(M,D,H,C), X, y)

    model = make_model(M,D,H,C)
    opt = AdamWOpt([], lr=1e-3, wd=1e-4)  # 파라미터는 trainer.compile에서 rebind
    if hasattr(opt, "ensure_initialized"): opt.ensure_initialized()
    if hasattr(opt, "set_grad_scale"):     opt.set_grad_scale(gs)

    trainer = CudaGraphTrainer(model, loss, opt)
    trainer.compile((M, D))

    L0, _ = loss.forward(model(X), y)
    trainer.one_step(X, y)
    L1, _ = loss.forward(model(X), y)

    print(f"[CHK] loss before={float(L0):.6f}, after={float(L1):.6f}")

if __name__ == "__main__":
    main()
