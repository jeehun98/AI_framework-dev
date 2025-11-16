# bench_gemm_binding.py
import os
import sys
import time

import numpy as np
import cupy as cp

# ============================================================
#  graph_executor_v2 패키지 import 준비
# ============================================================

# repo 루트: .../graph_executor_v2
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# python/ 디렉토리를 sys.path에 추가
PYTHON_ROOT = os.path.join(ROOT, "python")
if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)

from graph_executor_v2.ops import _ops_gemm as gemm


# ============================================================
#  벤치마크 유틸
# ============================================================

def _time_kernel(fn, iters=100, warmup=10):
    """cupy 이벤트 기반 평균 실행 시간(ms) 측정"""
    stream = cp.cuda.Stream.null

    # warmup
    for _ in range(warmup):
        fn()
    stream.synchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()

    elapsed_ms = cp.cuda.get_elapsed_time(start, end)  # total ms
    return elapsed_ms / iters


def bench_gemm_raw(
    m=1024,
    k=1024,
    n=1024,
    with_bias=True,
    act="relu",
    leaky_slope=0.01,
    iters=100,
    warmup=10,
):
    print("==== regemm GEMM+bias+act raw bench ====")
    print(f"shape: A({m},{k}) @ B({k},{n}) -> Y({m},{n})")
    print(f"act={act}, with_bias={with_bias}, iters={iters}, warmup={warmup}")

    rng = np.random.default_rng(2025)

    # -------- host → device 데이터 준비 --------
    A_h = rng.standard_normal((m, k), dtype=np.float32)
    B_h = rng.standard_normal((k, n), dtype=np.float32)
    gY_h = rng.standard_normal((m, n), dtype=np.float32)

    if with_bias:
        bias_h = rng.standard_normal((1, n), dtype=np.float32)
    else:
        bias_h = None

    A_d = cp.asarray(A_h)
    B_d = cp.asarray(B_h)
    gY_d = cp.asarray(gY_h)

    Y_d = cp.empty((m, n), dtype=cp.float32)
    Z_d = cp.empty((m, n), dtype=cp.float32)  # pre-act Z 저장용

    gA_d = cp.empty((m, k), dtype=cp.float32)
    gB_d = cp.empty((k, n), dtype=cp.float32)

    if with_bias:
        Bias_d = cp.asarray(bias_h)
        gBias_d = cp.empty((1, n), dtype=cp.float32)
        Bias_ptr = int(Bias_d.data.ptr)
        gBias_ptr = int(gBias_d.data.ptr)
    else:
        Bias_d = None
        gBias_d = None
        Bias_ptr = 0
        gBias_ptr = 0

    # C / gC 는 사용 안 함
    C_ptr = 0
    gC_ptr = 0

    # -------- 포인터 추출 --------
    A_ptr = int(A_d.data.ptr)
    B_ptr = int(B_d.data.ptr)
    Y_ptr = int(Y_d.data.ptr)
    Z_ptr = int(Z_d.data.ptr)
    gY_ptr = int(gY_d.data.ptr)
    gA_ptr = int(gA_d.data.ptr)
    gB_ptr = int(gB_d.data.ptr)

    # ============================================================
    #  Forward / Backward 래퍼
    # ============================================================

    def _fwd():
        gemm.forward_raw(
            A_ptr,
            B_ptr,
            Bias_ptr,
            Y_ptr,
            m,
            k,
            n,
            False,  # trans_a
            False,  # trans_b
            act,
            with_bias,
            leaky_slope,
            True,   # save_z
            Z_ptr,
            None,   # stream (default stream)
        )

    def _bwd():
        gemm.backward_raw(
            A_ptr,
            B_ptr,
            C_ptr,
            gY_ptr,
            Z_ptr,
            gA_ptr,
            gB_ptr,
            gC_ptr,
            gBias_ptr,
            m,
            k,
            n,
            False,  # trans_a
            False,  # trans_b
            act,
            with_bias,
            leaky_slope,
            None,   # stream
        )

    # 한 번 찍어보고 NaN 여부 정도만 확인 (레퍼런스 검증 없음)
    _fwd()
    _bwd()
    cp.cuda.runtime.deviceSynchronize()
    print("warm call done. (no correctness check, only NaN check)")

    Y_out = cp.asnumpy(Y_d)
    if np.isnan(Y_out).any():
        print("[warn] Y_out has NaN!")

    # ============================================================
    #  타이밍
    # ============================================================

    fwd_ms = _time_kernel(_fwd, iters=iters, warmup=warmup)
    bwd_ms = _time_kernel(_bwd, iters=iters, warmup=warmup)

    # FLOPs 계산
    # GEMM: 2 * M * K * N
    flops_gemm = 2.0 * m * k * n
    tflops_fwd = flops_gemm / (fwd_ms * 1e-3) / 1e12

    # backward_raw 내부:
    #   gA = gZ @ B^T (M,K,N)
    #   gB = A^T @ gZ (K,M,N)
    # → 대략 GEMM 2번 = 4 * M * K * N FLOPs 로 근사
    flops_bwd = 2.0 * flops_gemm
    tflops_bwd = flops_bwd / (bwd_ms * 1e-3) / 1e12

    print("---- result ----")
    print(f"forward:  {fwd_ms:8.4f} ms, ~{tflops_fwd:6.2f} TFLOPs (GEMM-only 기준)")
    print(f"backward: {bwd_ms:8.4f} ms, ~{tflops_bwd:6.2f} TFLOPs (GEMM×2 기준)")


if __name__ == "__main__":
    # 기본: 1024³, bias + ReLU
    bench_gemm_raw(
        m=1024,
        k=1024,
        n=1024,
        with_bias=True,
        act="relu",
        iters=100,
        warmup=10,
    )
