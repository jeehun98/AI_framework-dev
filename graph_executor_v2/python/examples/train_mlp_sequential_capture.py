# python/examples/train_mlp_sequential_capture.py
from __future__ import annotations
import os, sys
import math

# --- Import path (repo/python) ---
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
PKG  = os.path.join(ROOT, "python")
if PKG not in sys.path:
    sys.path.insert(0, PKG)


import cupy as cp

from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.ops import gemm as gops
from graph_executor_v2.losses.softmax_ce import SoftmaxCrossEntropy

def main():
    cp.random.seed(0)

    batch = 128
    in_dim = 256
    hid    = 256
    out_dim = 10
    steps = 200
    lr = 1e-3
    wd = 1e-2

    # 데이터
    X = cp.random.randn(batch, in_dim).astype(cp.float32)
    y = cp.random.randint(0, out_dim, size=(batch,), dtype=cp.int32)

    # 레이어 직접 구성(캡처-세이프 경로 사용하려면 내부 _into를 써야 하므로)
    l1 = Dense(hid, activation="relu", initializer="he",  use_native_bwd=True)
    l2 = Dense(out_dim, activation="none", initializer="xavier", use_native_bwd=True)
    l1.build((batch, in_dim))
    l2.build((batch, hid))

    # 손실
    loss_fn = SoftmaxCrossEntropy()

    # ---- 사전할당 버퍼들 (NO alloc)
    # Forward 버퍼
    z1 = cp.empty((batch, hid), dtype=cp.float32)        # pre-activation l1
    y1 = cp.empty((batch, hid), dtype=cp.float32)
    z2 = None                                            # l2는 act='none'이라 Z 불필요
    y2 = cp.empty((batch, out_dim), dtype=cp.float32)    # logits

    # Backward 버퍼
    dY = cp.empty_like(y2)                 # loss 도함수 wrt logits
    dx2 = cp.empty((batch, hid), dtype=cp.float32)
    dW2 = cp.empty((hid, out_dim), dtype=cp.float32)
    db2 = cp.empty((1, out_dim), dtype=cp.float32)

    dx1 = cp.empty((batch, in_dim), dtype=cp.float32)
    dW1 = cp.empty((in_dim, hid), dtype=cp.float32)
    db1 = cp.empty((1, hid), dtype=cp.float32)

    # GEMM 워크스페이스 (dZ 필수 / Lt ws 옵션)
    ws1 = gops.ensure_workspaces(batch, hid,  lt_bytes=(8<<20))
    ws2 = gops.ensure_workspaces(batch, out_dim, lt_bytes=(8<<20))

    # ---- 간단한 AdamW 수동 구현 (파라미터 수 적으니 직접)
    # 모멘트 버퍼
    mW1 = cp.zeros_like(l1.W); vW1 = cp.zeros_like(l1.W)
    mB1 = cp.zeros_like(l1.b); vB1 = cp.zeros_like(l1.b)
    mW2 = cp.zeros_like(l2.W); vW2 = cp.zeros_like(l2.W)
    mB2 = cp.zeros_like(l2.b); vB2 = cp.zeros_like(l2.b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    # ---- CUDA Graph 캡처
    stream = cp.cuda.Stream(non_blocking=True)
    g = cp.cuda.graph.Graph()
    with stream:
        # 더미 1 스텝 실행으로 shape/type warmup
        l1.forward_into(X, out=y1, z_out=z1, stream=stream.ptr)
        l2.forward_into(y1, out=y2, z_out=None, stream=stream.ptr)
        # loss + dY
        loss, dY_tmp = loss_fn.forward_with_grad(y2, y)
        dY[...] = dY_tmp

        # backward (l2 → l1)
        l2.backward_into(dY, gA_out=dx2, gW_out=dW2, gB_out=db2,
                         work_dZ=ws2.dZ, lt_workspace=ws2.lt_ws, stream=stream.ptr)
        l1.backward_into(dx2, gA_out=dx1, gW_out=dW1, gB_out=db1,
                         work_dZ=ws1.dZ, lt_workspace=ws1.lt_ws, stream=stream.ptr)

        # 간단 AdamW 적용(가중치 갱신)
        def adamw_step(W, dW, m, v):
            m[:] = beta1*m + (1-beta1)*dW
            v[:] = beta2*v + (1-beta2)*(dW*dW)
            m_hat = m/(1-beta1); v_hat = v/(1-beta2)
            W[:] = W - lr*(m_hat/(cp.sqrt(v_hat)+eps) + wd*W)

        adamw_step(l2.W, dW2, mW2, vW2); l2.b[:] = l2.b - lr*(db2 + wd*l2.b)
        adamw_step(l1.W, dW1, mW1, vW1); l1.b[:] = l1.b - lr*(db1 + wd*l1.b)

        # 그래프 캡처
        g.capture_begin(stream.ptr)
        # forward
        l1.forward_into(X, out=y1, z_out=z1, stream=stream.ptr)
        l2.forward_into(y1, out=y2, z_out=None, stream=stream.ptr)
        # loss+dY
        loss, dY_tmp = loss_fn.forward_with_grad(y2, y)
        dY[...] = dY_tmp
        # backward
        l2.backward_into(dY, gA_out=dx2, gW_out=dW2, gB_out=db2,
                         work_dZ=ws2.dZ, lt_workspace=ws2.lt_ws, stream=stream.ptr)
        l1.backward_into(dx2, gA_out=dx1, gW_out=dW1, gB_out=db1,
                         work_dZ=ws1.dZ, lt_workspace=ws1.lt_ws, stream=stream.ptr)
        # update
        adamw_step(l2.W, dW2, mW2, vW2); l2.b[:] = l2.b - lr*(db2 + wd*l2.b)
        adamw_step(l1.W, dW1, mW1, vW1); l1.b[:] = l1.b - lr*(db1 + wd*l1.b)
        g.capture_end(stream.ptr)

    gexec = cp.cuda.graph.GraphExec(g)

    # ---- 실행
    for step in range(1, steps+1):
        gexec.launch(stream.ptr)
        stream.synchronize()

        if step % 20 == 0:
            pred = y2.argmax(axis=1)
            acc = float((pred == y).mean())
            print(f"[{step:04d}] loss={float(loss):.4f}  acc={acc:.3f}")

    print("done.")

if __name__ == "__main__":
    main()
