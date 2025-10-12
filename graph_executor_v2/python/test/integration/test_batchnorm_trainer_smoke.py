# test/integration/test_batchnorm_trainer_smoke.py
import os, sys, math
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cupy as cp

from graph_executor_v2.layers.sequential import Sequential
from graph_executor_v2.layers.pad import Pad
from graph_executor_v2.layers.conv2d import Conv2D
from graph_executor_v2.layers.flatten import Flatten
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.layers.activations import ActivationLayer
from graph_executor_v2.losses.softmax_ce import SoftmaxCrossEntropy
from graph_executor_v2.optim.adamw import AdamWOpt
from graph_executor_v2.train.cuda_graph_trainer import CudaGraphTrainer

# BatchNorm2d 가 없을 수도 있으니 안전하게 처리
try:
    from graph_executor_v2.layers.batchnorm import BatchNorm2d
except Exception:
    BatchNorm2d = None


def make_model_with_bn(
    *,
    act: str = "relu",
    with_affine: bool = True,
    channels_last: bool = False,   # 현재 예제는 NCHW로만 사용
    N=8, Cin=3, H=16, W=16, hidden=32, classes=5
):
    """
    Conv → BN → Act → Flatten → Dense → Act → Dense
    - Conv/Dense는 activation="none"
    - BN은 2D 버전 (N,C,H,W 입력 가정)
    """
    if BatchNorm2d is None:
        raise RuntimeError("BatchNorm2d layer not available — skipping test.")

    # 활성화 레이어는 별도로 삽입 (BN의 활성화는 아님)
    act1 = ActivationLayer(act=act, save_y=True, name=f"Act1({act})")
    act2 = ActivationLayer(act=act, save_y=True, name=f"Act2({act})")

    m = Sequential(
        Pad(before=(1, 1), after=(1, 1), value=0.0),
        Conv2D(out_channels=8, kernel_size=3, padding=(0, 0), activation="none"),
        BatchNorm2d(eps=1e-5, momentum=0.1,
                    affine=with_affine, channels_last=channels_last, name="BN1"),
        act1,
        Conv2D(out_channels=8, kernel_size=3, padding=(1, 1), activation="none"),
        BatchNorm2d(eps=1e-5, momentum=0.1,
                    affine=with_affine, channels_last=channels_last, name="BN2"),
        act2,
        Flatten(),
        Dense(hidden, activation="none", initializer="he", use_native_bwd=True),
        ActivationLayer(act=act, save_y=True, name=f"Act3({act})"),
        Dense(classes, activation="none", initializer="xavier", use_native_bwd=True),
    )
    m.build((N, Cin, H, W))
    m.train(True)
    return m


def run_smoke_for_bn(config_name: str, *, with_affine: bool, channels_last: bool):
    N, Cin, H, W, C = 8, 3, 16, 16, 5
    cp.random.seed(2026)
    X = cp.random.randn(N, Cin, H, W).astype(cp.float32)
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    if BatchNorm2d is None:
        print(f"[SKIP] {config_name}: BatchNorm2d is not available.")
        return

    model = make_model_with_bn(
        act="relu",
        with_affine=with_affine,
        channels_last=channels_last,
        N=N, Cin=Cin, H=H, W=W, hidden=32, classes=C
    )

    loss = SoftmaxCrossEntropy()
    opt = AdamWOpt([], lr=1e-3, wd=1e-4)
    if hasattr(opt, "ensure_initialized"):
        opt.ensure_initialized()

    trainer = CudaGraphTrainer(model, loss, opt)
    trainer.compile((N, Cin, H, W))  # CUDA Graph capture

    last_L = None
    for t in range(3):
        L = trainer.one_step(X, y)
        print(f"[SMOKE][BN:{config_name}] step={t:02d} loss={L:.6f}")
        assert isinstance(L, float) and math.isfinite(L)
        last_L = L

    print(f"[OK] Integrated trainer smoke passed with BatchNorm2d({config_name}). Last loss={last_L:.6f}")


def main():
    print("== Integrated trainer smoke with BatchNorm2d inserted ==")
    # 두 가지 모드만 간단 확인 (Affine on/off). channels_last=False(NCHW)로 고정.
    cases = [
        ("affine", True, False),
        ("noaffine", False, False),
    ]
    for name, affine, ch_last in cases:
        run_smoke_for_bn(name, with_affine=affine, channels_last=ch_last)


if __name__ == "__main__":
    main()
