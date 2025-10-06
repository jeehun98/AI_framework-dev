# python/test/ops/test_gemm_backward.py
import os, sys, argparse
import numpy as np
import cupy as cp

# === Import path & CUDA DLL 경로 (Windows) ===
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", "..", ".."))
PKG  = os.path.join(ROOT, "python")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

cuda_bins = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
]
if hasattr(os, "add_dll_directory"):
    for d in cuda_bins:
        if os.path.isdir(d):
            os.add_dll_directory(d)

from graph_executor_v2.ops import require
ops_gemm = require("gemm")  # -> _ops_gemm


# ------------------- 유틸 -------------------
def list_all_pyd():
    roots = [
        os.path.join(ROOT, "python", "graph_executor_v2", "ops"),
        os.path.dirname(os.__file__),  # ...\Lib
    ]
    found = []
    for base in roots:
        for r, _, files in os.walk(base):
            for f in files:
                if f.startswith("_ops_gemm") and f.endswith(".pyd"):
                    found.append(os.path.join(r, f))
    for sp in sys.path:
        try:
            for r, _, files in os.walk(sp):
                for f in files:
                    if f.startswith("_ops_gemm") and f.endswith(".pyd"):
                        p = os.path.join(r, f)
                        if p not in found:
                            found.append(p)
        except Exception:
            pass
    return sorted(set(found))


def check_no_debug_string(pyd_path: str, needle=b"[BWD dbg]"):
    try:
        with open(pyd_path, "rb") as f:
            blob = f.read()
        return (needle not in blob)
    except Exception:
        return False


def finite_diff_check(A, B, bias, act="relu", eps=1e-3, tol=5e-2, seed=0):
    """
    간단 수치미분: Z = A@B + bias, Y = act(Z)
    임의 gY ~ N(0,1), analytic gA/gB/gBias vs finite-diff 비교.
    - 작은 텐서에서만 사용(시간 절약)
    """
    rng = np.random.default_rng(seed)
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    Z = A @ B + bias.reshape(1, N)
    gY = rng.standard_normal(size=(M, N)).astype(np.float32)

    out = ops_gemm.backward_numpy(A, B, gY, Z, act=act, bias_kind="pern", leaky_slope=0.0)
    gA, gB, gBias = out["gA"], out["gB"], out["gBias"]
    assert gA.shape == (M, K) and gB.shape == (K, N)

    def forward_with_act(A_, B_, bias_):
        Z_ = A_ @ B_ + bias_.reshape(1, N)
        if act == "relu":
            Y_ = np.maximum(Z_, 0)
        elif act == "leakyrelu":
            slope = 0.0
            Y_ = np.where(Z_ > 0, Z_, slope * Z_)
        elif act == "tanh":
            Y_ = np.tanh(Z_)
        elif act == "sigmoid":
            Y_ = 1 / (1 + np.exp(-Z_))
        elif act == "gelu":
            c = np.sqrt(2/np.pi)
            Y_ = 0.5 * Z_ * (1 + np.tanh(c*(Z_ + 0.044715*(Z_**3))))
        else:
            Y_ = Z_
        return Y_

    # gA finite-diff (한 원소만 샘플링)
    i, k = 0, 0
    A_pos = A.copy(); A_pos[i, k] += eps
    A_neg = A.copy(); A_neg[i, k] -= eps
    Y_pos = forward_with_act(A_pos, B, bias)
    Y_neg = forward_with_act(A_neg, B, bias)
    loss_pos = (Y_pos * gY).sum()
    loss_neg = (Y_neg * gY).sum()
    gA_fd = (loss_pos - loss_neg) / (2*eps)
    err_gA = abs(gA_fd - gA[i, k]) / (abs(gA_fd) + 1e-6)

    # gB finite-diff
    k_, j = 0, 0
    B_pos = B.copy(); B_pos[k_, j] += eps
    B_neg = B.copy(); B_neg[k_, j] -= eps
    Y_pos = forward_with_act(A, B_pos, bias)
    Y_neg = forward_with_act(A, B_neg, bias)
    loss_pos = (Y_pos * gY).sum()
    loss_neg = (Y_neg * gY).sum()
    gB_fd = (loss_pos - loss_neg) / (2*eps)
    err_gB = abs(gB_fd - gB[k_, j]) / (abs(gB_fd) + 1e-6)

    # gBias finite-diff (perN 가정)
    j_ = 0
    bias_pos = bias.copy(); bias_pos[j_] += eps
    bias_neg = bias.copy(); bias_neg[j_] -= eps
    Y_pos = forward_with_act(A, B, bias_pos)
    Y_neg = forward_with_act(A, B, bias_neg)
    loss_pos = (Y_pos * gY).sum()
    loss_neg = (Y_neg * gY).sum()
    gBias_fd = (loss_pos - loss_neg) / (2*eps)
    err_gBias = abs(gBias_fd - (gBias[j_] if gBias is not None else 0.0)) / (abs(gBias_fd) + 1e-6)

    ok = (err_gA < tol) and (err_gB < tol) and (err_gBias < tol)
    return ok, dict(err_gA=float(err_gA), err_gB=float(err_gB), err_gBias=float(err_gBias))


# ------------------- Z save 트리아지 -------------------
Device   = ops_gemm.Device
DType    = ops_gemm.DType
Layout   = ops_gemm.Layout
make2d   = ops_gemm.make_tensor_2d

def make2d_from_cupy(arr_cp):
    """CuPy 배열 -> Tensor 래핑 (동일 디바이스 메모리 사용)"""
    assert arr_cp.flags['C_CONTIGUOUS'], "CuPy array must be C-contiguous"
    ptr   = int(arr_cp.data.ptr)
    shape = tuple(int(x) for x in arr_cp.shape)
    return make2d(ptr, shape, dtype=DType.F32, device=Device.CUDA, device_index=0)

def best_match(a: np.ndarray, b: np.ndarray):
    """
    a, b에 대해 (same, aT, bT, bothT) 중 shape가 맞는 조합만 평가,
    최소 max_abs와 태그를 반환.
    """
    cands = []
    if a.shape == b.shape:
        cands.append(("same", a, b))
    if a.T.shape == b.shape:
        cands.append(("aT", a.T, b))
    if a.shape == b.T.shape:
        cands.append(("bT", a, b.T))
    if a.T.shape == b.T.shape:
        cands.append(("bothT", a.T, b.T))
    if not cands:
        raise AssertionError(f"no shape-compatible orientation: a{a.shape} vs b{b.shape}")

    def mabs(x, y): return float(np.max(np.abs(x - y)))
    errs = [(tag, mabs(x, y)) for tag, x, y in cands]
    tag_min, err_min = min(errs, key=lambda t: t[1])
    return err_min, tag_min, dict(errs=errs)

def run_case(A, B, bias, act: str, with_bias: bool, strict: bool = False, atol: float = 2e-4):
    print(f"\n=== CASE: act={act}, with_bias={with_bias} ===")
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    # 1) GPU 버퍼
    A_d     = cp.asarray(A, dtype=cp.float32, order='C')
    B_d     = cp.asarray(B, dtype=cp.float32, order='C')
    Bias_d  = cp.asarray(bias.reshape(1, N), dtype=cp.float32, order='C') if with_bias else None
    Y_d     = cp.empty((M, N), dtype=cp.float32, order='C')
    Zsave_d = cp.empty((M, N), dtype=cp.float32, order='C')
    Zsave_d.fill(-777.0)  # 초기 패턴

    # 2) 래핑
    A_t     = make2d_from_cupy(A_d)
    B_t     = make2d_from_cupy(B_d)
    Bias_t  = make2d_from_cupy(Bias_d) if with_bias else None
    Y_t     = make2d_from_cupy(Y_d)
    Zsave_t = make2d_from_cupy(Zsave_d)

    # 3) 호출 (save_z=True)
    ops_gemm.forward_ex(
        A_t, B_t, Bias_t, Y_t,
        False, False, act, with_bias, 0.01,
        True,        # save_z
        Zsave_t,     # Z_saved
        None
    )
    cp.cuda.Stream.null.synchronize()

    # 4) 호스트 비교용
    Y_gpu = cp.asnumpy(Y_d)
    Zsv   = cp.asnumpy(Zsave_d)
    Z_pre = (A @ B)
    if with_bias:
        Z_pre = Z_pre + bias.reshape(1, N)

    # post-activation
    if act == "relu":
        Y_np = np.maximum(Z_pre, 0)
    elif act == "leakyrelu":
        slope = 0.01
        Y_np = np.where(Z_pre > 0, Z_pre, slope * Z_pre)
    elif act == "tanh":
        Y_np = np.tanh(Z_pre)
    elif act == "sigmoid":
        Y_np = 1 / (1 + np.exp(-Z_pre))
    elif act == "gelu":
        c = np.sqrt(2/np.pi)
        Y_np = 0.5 * Z_pre * (1 + np.tanh(c*(Z_pre + 0.044715*(Z_pre**3))))
    else:  # "none"
        Y_np = Z_pre

    # 5) 지표 출력 + (선택) 엄격 검증
    y_err, _, _ = best_match(Y_gpu, Y_np)  # 정상이라면 'same'에서 0 근처
    print("max_abs Y(gpu) vs Y(calc):", y_err)

    pre_err,  pre_tag,  pre_all  = best_match(Zsv, Z_pre)
    post_err, post_tag, post_all = best_match(Zsv, Y_np)

    wrote = not np.allclose(Zsv, -777.0)

    print("Z_saved ~ Z(pre):  err=", pre_err,  " tag=", pre_tag,  " cand=", pre_all["errs"])
    print("Z_saved ~ Y(post): err=", post_err, " tag=", post_tag, " cand=", post_all["errs"])
    print("Z_saved_written?:", wrote)

    if strict:
        # act="none"인 경우엔 설계상 pre 저장이 정석(문서 기준)
        if act == "none":
            assert y_err < atol, f"Y mismatch too large: {y_err}"
            assert wrote, "Z_saved was not written"
            # 전치 없이 같은 레이아웃('same')으로 맞는 것을 기대
            assert pre_tag == "same" and pre_err < atol, \
                f"expect Z_saved == Z(pre) (same), got tag={pre_tag}, err={pre_err}"

    return dict(
        y_err=y_err,
        pre_err=pre_err, pre_tag=pre_tag,
        post_err=post_err, post_tag=post_tag,
        wrote=wrote
    )

# --- 추가 진단: bias 1D 시나리오 ---
def run_case_bias_1d(A, B, bias_1d, act: str, strict=False, atol: float = 2e-4):
    """
    API가 2D만 허용하므로 (N,)를 (1,N)으로 reshape하여 make_tensor_2d에 전달.
    런처 쪽에서 shape 기반으로 PerN을 판정하는지 진단한다.
    """
    print(f"\n=== CASE: act={act}, with_bias=True, bias=(N,) 1D (wrapped as (1,N)) ===")
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    assert bias_1d.shape == (N,)

    # 1) 디바이스 버퍼
    A_d = cp.asarray(A, dtype=cp.float32, order='C')
    B_d = cp.asarray(B, dtype=cp.float32, order='C')
    Bias1d_row_d = cp.asarray(bias_1d.reshape(1, N), dtype=cp.float32, order='C')  # (1,N)
    Y_d = cp.empty((M, N), dtype=cp.float32, order='C')
    Zsave_d = cp.empty((M, N), dtype=cp.float32, order='C'); Zsave_d.fill(-777.)

    # 2) 래핑 (모두 2D)
    A_t = make2d_from_cupy(A_d)
    B_t = make2d_from_cupy(B_d)
    Bias1d_row_t = make2d_from_cupy(Bias1d_row_d)
    Y_t = make2d_from_cupy(Y_d)
    Zsave_t = make2d_from_cupy(Zsave_d)

    # 3) 호출
    ops_gemm.forward_ex(
        A_t, B_t, Bias1d_row_t, Y_t,
        False, False, act, True, 0.01,
        True, Zsave_t, None
    )
    cp.cuda.Stream.null.synchronize()

    # 4) 비교
    Zsv = cp.asnumpy(Zsave_d)
    Y_gpu = cp.asnumpy(Y_d)

    Z_pre = (A @ B) + bias_1d.reshape(1, N)
    if act == "none":
        Y_np = Z_pre
    else:
        # 필요시 다른 액티베이션도 확대
        Y_np = np.maximum(Z_pre, 0) if act == "relu" else Z_pre

    z_err, z_tag, _ = best_match(Zsv, Z_pre)
    y_err, y_tag, _ = best_match(Y_gpu, Y_np)

    print("Z_saved vs Z(pre):", z_err, " tag:", z_tag)
    print("Y(gpu)  vs Y(calc):", y_err, " tag:", y_tag)

    if strict and act == "none":
        assert z_tag == "same" and z_err < atol, f"Z_saved!=Z(pre) for 1D bias->(1,N); err={z_err}, tag={z_tag}"
        assert y_err < atol, f"Y mismatch with 1D bias->(1,N); err={y_err}"


# ------------------- 메인 -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finite-diff", action="store_true", help="작은 텐서에서 수치 미분 검증 수행")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strict", action="store_true", help="저수준 save_z 케이스에 대해 엄격 검증")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # === 모듈 로드 경로 확인 ===
    print("LOADED:", ops_gemm.__file__)

    # === 중복 pyd 탐지 ===
    all_pyd = list_all_pyd()
    if all_pyd:
        print("FOUND_PYDS:")
        for p in all_pyd:
            mark = " <-- LOADED" if os.path.abspath(p) == os.path.abspath(ops_gemm.__file__) else ""
            print("  ", p, mark)

    # === 로드된 pyd에 디버그 문자열이 남았는지 검사 ===
    ok_no_dbg = check_no_debug_string(ops_gemm.__file__)
    print("BINARY_HAS_NO_[BWD dbg]:", ok_no_dbg)
    assert ok_no_dbg, "Loaded pyd still contains [BWD dbg] string!"

    # === 랜덤 텐서 준비 ===
    M, K, N = 8, 7, 5
    A = rng.standard_normal(size=(M, K), dtype=np.float32)
    B = rng.standard_normal(size=(K, N), dtype=np.float32)
    bias = rng.standard_normal(size=(N,), dtype=np.float32)
    print("max|bias|:", float(np.max(np.abs(bias))))  # bias 미적용 의심치 확인용

    # ------------------------------------------------------------------
    # (A) 고수준: NumPy 편의 경로로 forward/backward 기본 동작 확인
    # ------------------------------------------------------------------
    Y = ops_gemm.forward_numpy(A, B, bias, act="relu", leaky_slope=0.0)
    print("Y.shape:", Y.shape)
    assert Y.shape == (M, N)

    gY = rng.standard_normal(size=(M, N), dtype=np.float32)
    Z  = (A @ B) + bias.reshape(1, N)

    out = ops_gemm.backward_numpy(A, B, gY, Z, act="relu", bias_kind="pern", leaky_slope=0.0)
    print("BACKWARD_KEYS:", list(out.keys()))
    assert out["gC"] is None

    gA, gB, gBias = out["gA"], out["gB"], out["gBias"]
    print("gA.shape:", gA.shape, "gB.shape:", gB.shape)
    if gBias is not None:
        print("gBias.shape:", gBias.shape)
        assert gBias.shape == (N,)

    assert gA.shape == (M, K)
    assert gB.shape == (K, N)

    # === 선택: 수치 미분 체크 ===
    if args.finite_diff:
        print("Running finite-diff check...(small tensors)")
        M2, K2, N2 = 4, 3, 3
        A2 = rng.standard_normal(size=(M2, K2), dtype=np.float32)
        B2 = rng.standard_normal(size=(K2, N2), dtype=np.float32)
        bias2 = rng.standard_normal(size=(N2,), dtype=np.float32)
        ok, errs = finite_diff_check(A2, B2, bias2, act="relu", eps=1e-3, tol=8e-2, seed=args.seed+1)
        print("FINITE_DIFF_OK:", ok, errs)
        assert ok, f"Finite-diff mismatch: {errs}"

    # ------------------------------------------------------------------
    # (B) 저수준: save_z=True 경로 진단 (전치/바이어스/포인터/저장위치)
    # ------------------------------------------------------------------
    # 1) 순수 GEMM (bias 없음/act 없음) -> Z_saved == A@B가 정상
    run_case(A, B, bias, act="none", with_bias=False, strict=args.strict)

    # 2) GEMM+bias (act 없음) -> Z_saved == A@B + bias (여기서 지금 문제가 보일 가능성 높음)
    run_case(A, B, bias, act="none", with_bias=True, strict=args.strict)

    # 3) GEMM+bias+ReLU -> 구현 정책에 따라 pre/post 중 하나와 일치할 수 있음
    run_case(A, B, bias, act="relu", with_bias=True, strict=False)

    # ------------------------------------------------------------------
    # (C) bias 1D 시나리오도 추가 확인 (PerN 1D 지원 여부)
    # ------------------------------------------------------------------
    run_case_bias_1d(A, B, bias, act="none", strict=args.strict)
    run_case_bias_1d(A, B, bias, act="relu", strict=False)

    print("DONE: forward/backward + save_z + bias-kind triage complete.")


if __name__ == "__main__":
    main()
