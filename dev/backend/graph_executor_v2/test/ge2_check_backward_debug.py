# ge2_check_backward_debug.py
import numpy as np
import cupy as cp
import graph_executor_v2 as ge2

# ---------- 활성/도함수 (레퍼런스) ----------
def gelu(x):
    k0 = np.sqrt(2.0/np.pi)
    k1 = 0.044715
    t = k0*(x + k1*x*x*x)
    return 0.5*x*(1.0 + np.tanh(t))
def dgelu(x):
    c  = np.sqrt(2.0/np.pi)
    k1 = 0.044715
    x2 = x*x
    t  = c*(x + k1*x*x2)
    th = np.tanh(t)
    sech2 = 1.0 - th*th
    dt = c*(1.0 + 3.0*k1*x2)
    return 0.5*(1.0 + th) + 0.5*x*sech2*dt

def act(x, kind, a=0.01):
    if kind == getattr(ge2.ActKind, "None"):    return x
    if kind == ge2.ActKind.ReLU:    return np.maximum(x, 0)
    if kind == ge2.ActKind.LeakyReLU: return np.where(x>0, x, a*x)
    if kind == ge2.ActKind.GELU:    return gelu(x)
    if kind == ge2.ActKind.Sigmoid: return 1/(1+np.exp(-x))
    if kind == ge2.ActKind.Tanh:    return np.tanh(x)
    raise ValueError
def dact(x, kind, a=0.01):
    if kind == getattr(ge2.ActKind, "None"):    return np.ones_like(x)
    if kind == ge2.ActKind.ReLU:    return (x>0).astype(x.dtype)
    if kind == ge2.ActKind.LeakyReLU: return np.where(x>0, 1, a).astype(x.dtype)
    if kind == ge2.ActKind.GELU:    return dgelu(x)
    if kind == ge2.ActKind.Sigmoid:
        s = 1/(1+np.exp(-x)); return s*(1-s)
    if kind == ge2.ActKind.Tanh:
        t = np.tanh(x); return 1 - t*t
    raise ValueError

def max_abs_diff(a, b):
    return float(np.max(np.abs(a-b)))

def run_case(M=64, N=48, K=32,
             act_kind=ge2.ActKind.GELU,
             bias_kind=ge2.BiasKind.PerN,
             use_C=True, save_preact=True,
             alpha=1.1, beta=0.9, leaky=0.02, seed=0):

    rng = np.random.default_rng(seed)
    hA = rng.standard_normal((M,K), np.float32)
    hB = rng.standard_normal((K,N), np.float32)
    hC = rng.standard_normal((M,N), np.float32)
    hBias = None
    if bias_kind == ge2.BiasKind.Scalar:
        hBias = np.array([0.1], dtype=np.float32)
    elif bias_kind == ge2.BiasKind.PerM:
        hBias = np.full((M,), 0.1, np.float32)
    elif bias_kind == ge2.BiasKind.PerN:
        hBias = np.full((N,), 0.1, np.float32)

    # --- GPU tensors
    A = cp.asarray(hA); B = cp.asarray(hB)
    C = cp.asarray(hC) if use_C else None
    D = cp.empty((M,N), np.float32)
    Z = cp.empty((M,N), np.float32) if save_preact else None
    bias = cp.asarray(hBias) if hBias is not None else None

    # --- Forward(EX)
    px = ge2.GemmBiasActParamsEx()
    px.M, px.N, px.K = M,N,K
    px.lda, px.ldb, px.ldc, px.ldd = K, N, N, N
    px.alpha, px.beta = float(alpha), float(beta)
    px.use_C = 1 if use_C else 0
    px.has_bias = 1 if (bias is not None) else 0
    px.bias_kind = bias_kind
    px.act_kind  = act_kind
    px.leaky_slope = float(leaky)
    px.save_preact = 1 if save_preact else 0
    px.ldZ = 0  # 0이면 ldd 사용

    ge2.gemm_bias_act_f32_ex(
        int(A.data.ptr), int(B.data.ptr),
        int(C.data.ptr) if C is not None else None,
        int(D.data.ptr),
        int(bias.data.ptr) if bias is not None else None,
        int(Z.data.ptr) if Z is not None else None,
        px, None
    )

    # --- CPU ref forward
    hAB  = hA @ hB
    hPre = alpha*hAB + (beta*hC if use_C else 0)
    if bias_kind == ge2.BiasKind.Scalar and hBias is not None:
        hPre = hPre + hBias[0]
    elif bias_kind == ge2.BiasKind.PerM and hBias is not None:
        hPre = hPre + hBias[:, None]
    elif bias_kind == ge2.BiasKind.PerN and hBias is not None:
        hPre = hPre + hBias[None, :]
    hDref = act(hPre, act_kind, leaky)

    hD = cp.asnumpy(D)
    fwd_diff = max_abs_diff(hD, hDref)

    # --- Backward 준비
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

    # ---- Debug: BWD 출력버퍼 초기화
    gA.fill(0); gB.fill(0)
    if gC is not None: gC.fill(0)
    if gBias is not None: gBias.fill(0)

    pb = ge2.GemmBiasActBwdParams()
    pb.M, pb.N, pb.K = M,N,K
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
        int(Z.data.ptr) if Z is not None else int(cp.asarray(hPre).data.ptr),
        int(gA.data.ptr), int(gB.data.ptr),
        int(gC.data.ptr) if gC is not None else None,
        int(gBias.data.ptr) if gBias is not None else None,
        pb, None
    )

    # --- CPU ref backward
    hZ  = hPre
    hGZ = hGY * dact(hZ, act_kind, leaky)

    hGA = hGZ @ hB.T
    hGB = hA.T @ hGZ
    hGC = beta*hGZ if use_C else None

    if bias_kind == ge2.BiasKind.Scalar:
        hGBias = np.array([np.sum(hGZ, dtype=np.float32)], dtype=np.float32)
    elif bias_kind == ge2.BiasKind.PerM:
        hGBias = np.sum(hGZ, axis=1, dtype=np.float32)
    elif bias_kind == ge2.BiasKind.PerN:
        hGBias = np.sum(hGZ, axis=0, dtype=np.float32)
    else:
        hGBias = None

    # --- 비교
    dA = cp.asnumpy(gA); dB = cp.asnumpy(gB)
    dA_diff = max_abs_diff(dA, hGA)
    dB_diff = max_abs_diff(dB, hGB)
    if use_C:
        dC_diff = max_abs_diff(cp.asnumpy(gC), hGC)
    else:
        dC_diff = 0.0
    if gBias is not None:
        dBias_diff = max_abs_diff(cp.asnumpy(gBias), hGBias)
    else:
        dBias_diff = 0.0

    print("="*60)
    print(f"[Case] M={M},N={N},K={K}, act={act_kind}, bias={bias_kind}, use_C={use_C}")
    print(f"[FWD] max|D-Dref| = {fwd_diff:.8f}")
    print(f"[BWD] dA max|diff| = {dA_diff:.8f}")
    print(f"[BWD] dB max|diff| = {dB_diff:.8f}")
    if use_C: print(f"[BWD] dC max|diff| = {dC_diff:.8f}")
    if gBias is not None: print(f"[BWD] dBias max|diff| = {dBias_diff:.8f}")
    print(f"Shapes: A{hA.shape}, B{hB.shape}, C{hC.shape}")
    if hBias is not None:
        print(f"Bias shape: {hBias.shape}, kind={bias_kind}")
    print(f"alpha={alpha}, beta={beta}, leaky={leaky}")
    print("="*60)

    tol = 3e-4
    ok = (fwd_diff<tol and dA_diff<tol and dB_diff<tol and 
          (not use_C or dC_diff<tol) and (gBias is None or dBias_diff<tol))
    print(f"[RESULT] OK={ok} (tol={tol:.1e})")
    return ok

if __name__ == "__main__":
    run_case(M=64, N=48, K=32,
             act_kind=ge2.ActKind.GELU,
             bias_kind=ge2.BiasKind.PerN,
             use_C=True, save_preact=True)

    run_case(M=64, N=64, K=32,
             act_kind=ge2.ActKind.ReLU,
             bias_kind=ge2.BiasKind.Scalar,
             use_C=False, save_preact=True)

    run_case(M=96, N=80, K=40,
             act_kind=ge2.ActKind.LeakyReLU,
             bias_kind=ge2.BiasKind.PerM,
             use_C=True, save_preact=True, leaky=0.02)
