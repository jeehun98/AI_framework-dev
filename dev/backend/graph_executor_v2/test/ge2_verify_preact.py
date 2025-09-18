# ge2_verify_preact.py  (BWD 마스크/노름 진단 강화판)
import numpy as np
import cupy as cp
import graph_executor_v2 as ge2

def gelu(x):
    k0 = np.sqrt(2.0/np.pi); k1 = 0.044715
    t = k0*(x + k1*x*x*x)
    return 0.5*x*(1.0 + np.tanh(t))

def act(x, kind, a=0.01):
    if kind == getattr(ge2.ActKind, "None"):    return x
    if kind == ge2.ActKind.ReLU:    return np.maximum(x, 0)
    if kind == ge2.ActKind.LeakyReLU: return np.where(x>0, x, a*x)
    if kind == ge2.ActKind.GELU:    return gelu(x)
    if kind == ge2.ActKind.Sigmoid: return 1/(1+np.exp(-x))
    if kind == ge2.ActKind.Tanh:    return np.tanh(x)
    raise ValueError

def make_bias(N, M, kind, val=0.1):
    if kind == ge2.BiasKind.Scalar:
        return np.array([val], np.float32)
    if kind == ge2.BiasKind.PerN:
        return np.full((N,), val, np.float32)
    if kind == ge2.BiasKind.PerM:
        return np.full((M,), val, np.float32)
    return None

def relu_mask_stats(hZ):
    pos = (hZ>0).sum()
    zeros = (hZ==0).sum()
    neg = (hZ<0).sum()
    return pos, zeros, neg

def norms(tag, arr):
    a = arr.ravel()
    return f"{tag}: L2={float(np.linalg.norm(a)):.6f}, L1={float(np.sum(np.abs(a))):.6f}, max={float(np.max(np.abs(a))):.6f}"

def run_forward_and_check(M, N, K, act_kind, bias_kind,
                          use_C, alpha=1.1, beta=0.9, leaky=0.02, seed=0):
    print("="*70)
    print(f"[CHECK] M={M},N={N},K={K}, act={act_kind}, bias={bias_kind}, use_C={use_C}")
    rng = np.random.default_rng(seed)
    hA = rng.standard_normal((M,K), np.float32)
    hB = rng.standard_normal((K,N), np.float32)
    hC = rng.standard_normal((M,N), np.float32) if use_C else None
    hBias = make_bias(N, M, bias_kind)

    # --- GPU bufs
    A = cp.asarray(hA); B = cp.asarray(hB)
    C = cp.asarray(hC) if hC is not None else None
    D = cp.empty((M,N), np.float32)
    Z = cp.empty((M,N), np.float32)
    bias = cp.asarray(hBias) if hBias is not None else None

    # --- Forward(EX) with Z stash
    px = ge2.GemmBiasActParamsEx()
    px.M, px.N, px.K = M, N, K
    px.lda, px.ldb, px.ldc, px.ldd = K, N, N, N
    px.alpha, px.beta = float(alpha), float(beta)
    px.use_C = 1 if use_C else 0
    px.has_bias = 1 if (bias is not None) else 0
    px.bias_kind = bias_kind
    px.act_kind  = act_kind
    px.leaky_slope = float(leaky)
    px.save_preact = 1
    px.ldZ = 0  # 0 -> use ldd

    ge2.gemm_bias_act_f32_ex(
        int(A.data.ptr), int(B.data.ptr),
        int(C.data.ptr) if C is not None else None,
        int(D.data.ptr),
        int(bias.data.ptr) if bias is not None else None,
        int(Z.data.ptr),
        px, None
    )

    # --- CPU pre / act for reference
    hAB = hA @ hB
    hPre = alpha*hAB + (beta*hC if use_C else 0)
    if bias_kind == ge2.BiasKind.Scalar and hBias is not None:
        hPre = hPre + hBias[0]
    elif bias_kind == ge2.BiasKind.PerM and hBias is not None:
        hPre = hPre + hBias[:, None]
    elif bias_kind == ge2.BiasKind.PerN and hBias is not None:
        hPre = hPre + hBias[None, :]
    hDref = act(hPre, act_kind, leaky)

    # --- Compare Z vs pre, D vs Dref
    hZ = cp.asnumpy(Z)
    hD = cp.asnumpy(D)
    z_pre_diff = float(np.max(np.abs(hZ - hPre)))
    d_diff = float(np.max(np.abs(hD - hDref)))
    print(f"max|Z - pre| = {z_pre_diff:.8f}")
    print(f"max|D - Dref| = {d_diff:.8f}")

    # --- Prepare BWD inputs
    hGY = rng.standard_normal((M,N), np.float32)
    gY = cp.asarray(hGY)
    gA = cp.empty((M,K), np.float32)
    gB = cp.empty((K,N), np.float32)
    gC = cp.empty((M,N), np.float32) if use_C else None
    if bias_kind == ge2.BiasKind.Scalar:
        gBias = cp.empty((1,), np.float32)
    elif bias_kind == ge2.BiasKind.PerM:
        gBias = cp.empty((M,), np.float32)
    elif bias_kind == ge2.BiasKind.PerN:
        gBias = cp.empty((N,), np.float32)
    else:
        gBias = None

    # --- BWD 전 진단(Host) : ReLU면 마스크 통계 출력
    print(norms("||Z||", hZ), norms("||gY||", hGY))
    if act_kind == ge2.ActKind.ReLU:
        pos, zeros, neg = relu_mask_stats(hZ)
        total = hZ.size
        print(f"ReLU mask stats: pos={pos} ({pos/total:.2%}), zeros={zeros} ({zeros/total:.2%}), neg={neg} ({neg/total:.2%})")

    # --- Backward(EX) using the same Z
    pb = ge2.GemmBiasActBwdParams()
    pb.M, pb.N, pb.K = M, N, K
    pb.lda, pb.ldb, pb.ldc = K, N, N
    pb.ldgY, pb.ldZ = N, N
    pb.ldgA, pb.ldgB = K, N
    pb.ldgC = N
    pb.alpha, pb.beta = float(alpha), float(beta)
    pb.bias_kind, pb.act_kind = bias_kind, act_kind
    pb.leaky_slope = float(leaky)

    ge2.gemm_bias_act_bwd_f32_ex(
        int(A.data.ptr), int(B.data.ptr),
        int(C.data.ptr) if C is not None else None,
        int(gY.data.ptr),
        int(Z.data.ptr),  # forward Z 그대로
        int(gA.data.ptr), int(gB.data.ptr),
        int(gC.data.ptr) if gC is not None else None,
        int(gBias.data.ptr) if gBias is not None else None,
        pb, None
    )

    # --- CPU ref BWD
    def dact(x, kind, a=0.01):
        if kind == getattr(ge2.ActKind, "None"):    return np.ones_like(x, np.float32)
        if kind == ge2.ActKind.ReLU:    return (x>0).astype(np.float32)
        if kind == ge2.ActKind.LeakyReLU: return np.where(x>0, 1, a).astype(np.float32)
        if kind == ge2.ActKind.GELU:
            c  = np.sqrt(2.0/np.pi); k1 = 0.044715
            x2 = x*x; t  = c*(x + k1*x*x2); th = np.tanh(t)
            sech2 = 1.0 - th*th; dt = c*(1.0 + 3.0*k1*x2)
            return 0.5*(1.0 + th) + 0.5*x*sech2*dt
        if kind == ge2.ActKind.Sigmoid:
            s = 1/(1+np.exp(-x)); return s*(1-s)
        if kind == ge2.ActKind.Tanh:
            t = np.tanh(x); return 1 - t*t
        raise ValueError

    hMask = dact(hPre, act_kind, leaky)
    hGZ   = hGY * hMask
    hGA = (hGZ @ hB.T) * alpha
    hGB = (hA.T @ hGZ) * alpha
    hGC = beta*hGZ if use_C else None
    if bias_kind == ge2.BiasKind.Scalar:
        hGBias = np.array([np.sum(hGZ, dtype=np.float32)], np.float32)
    elif bias_kind == ge2.BiasKind.PerM:
        hGBias = np.sum(hGZ, axis=1, dtype=np.float32)
    elif bias_kind == ge2.BiasKind.PerN:
        hGBias = np.sum(hGZ, axis=0, dtype=np.float32)
    else:
        hGBias = None

    dA = cp.asnumpy(gA); dB = cp.asnumpy(gB)
    print(norms("||GZ_ref||", hGZ))
    print(norms("||dA_gpu||", dA), norms("||dA_ref||", hGA))
    print(norms("||dB_gpu||", dB), norms("||dB_ref||", hGB))

    dA_diff = float(np.max(np.abs(dA - hGA)))
    dB_diff = float(np.max(np.abs(dB - hGB)))
    print(f"max|dA - dAref| = {dA_diff:.8f}")
    print(f"max|dB - dBref| = {dB_diff:.8f}")
    if use_C:
        dC_gpu = cp.asnumpy(gC)
        print(norms("||dC_gpu||", dC_gpu), norms("||dC_ref||", hGC))
        dC_diff = float(np.max(np.abs(dC_gpu - hGC)))
        print(f"max|dC - dCref| = {dC_diff:.8f}")
    if gBias is not None:
        gBias_gpu = cp.asnumpy(gBias)
        print(norms("||gBias_gpu||", gBias_gpu), norms("||gBias_ref||", hGBias))
        dBias_diff = float(np.max(np.abs(gBias_gpu - hGBias)))
        print(f"max|dBias - dBiasRef| = {dBias_diff:.8f}")

    # 헤드 비교
    print("-"*70)
    print("Head dumps:")
    print("Z[0:4,0:6]:\n", hZ[:4,:6])
    print("pre[0:4,0:6]:\n", hPre[:4,:6])
    print("mask[0:4,0:6]:\n", hMask[:4,:6])
    print("="*70)

if __name__ == "__main__":
    # ① ReLU + Scalar + no C
    run_forward_and_check(
        M=64, N=64, K=32,
        act_kind=ge2.ActKind.ReLU,
        bias_kind=ge2.BiasKind.Scalar,
        use_C=False,
        alpha=1.1, beta=0.9, leaky=0.02, seed=0
    )

    # ② LeakyReLU + PerM + use_C
    run_forward_and_check(
        M=96, N=80, K=40,
        act_kind=ge2.ActKind.LeakyReLU,
        bias_kind=ge2.BiasKind.PerM,
        use_C=True,
        alpha=1.1, beta=0.9, leaky=0.02, seed=1
    )
