# ge2_check_backward_probe.py
import numpy as np
import cupy as cp
import graph_executor_v2 as ge2

# ----- reference activations (same as before) -----
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

def act_fn(kind):
    def _relu(x): return np.maximum(x, 0)
    def _leaky(x,a): return np.where(x>0, x, a*x)
    def _sigmoid(x): return 1/(1+np.exp(-x))
    def _tanh(x): return np.tanh(x)
    if kind == "None":
        return lambda x, a=0.01: x
    if kind == "ReLU":
        return lambda x, a=0.01: _relu(x)
    if kind == "LeakyReLU":
        return lambda x, a=0.01: _leaky(x,a)
    if kind == "GELU":
        return lambda x, a=0.01: gelu(x)
    if kind == "Sigmoid":
        return lambda x, a=0.01: _sigmoid(x)
    if kind == "Tanh":
        return lambda x, a=0.01: _tanh(x)
    raise ValueError

def dact_fn(kind):
    def _drelu(x,a): return (x>0).astype(x.dtype)
    def _dleaky(x,a): return np.where(x>0, 1, a).astype(x.dtype)
    def _dsigmoid(x,a):
        s = 1/(1+np.exp(-x)); return s*(1-s)
    def _dtanh(x,a):
        t=np.tanh(x); return 1-t*t
    if kind == "None":
        return lambda x,a: np.ones_like(x)
    if kind == "ReLU":
        return lambda x,a: _drelu(x,a)
    if kind == "LeakyReLU":
        return lambda x,a: _dleaky(x,a)
    if kind == "GELU":
        return lambda x,a: dgelu(x)
    if kind == "Sigmoid":
        return lambda x,a: _dsigmoid(x,a)
    if kind == "Tanh":
        return lambda x,a: _dtanh(x,a)
    raise ValueError

def max_abs_diff(a,b): return float(np.max(np.abs(a-b)))

def probe_case(M, N, K,
               act_kind, bias_kind,
               use_C, save_preact,
               alpha, beta, leaky, seed=0):
    rng = np.random.default_rng(seed)
    hA = rng.standard_normal((M,K), np.float32)
    hB = rng.standard_normal((K,N), np.float32)
    hC = rng.standard_normal((M,N), np.float32)
    # bias host
    hBias = None
    if bias_kind == ge2.BiasKind.Scalar: hBias = np.array([0.1], np.float32)
    elif bias_kind == ge2.BiasKind.PerM: hBias = np.full((M,), 0.1, np.float32)
    elif bias_kind == ge2.BiasKind.PerN: hBias = np.full((N,), 0.1, np.float32)

    # GPU tensors
    A=cp.asarray(hA); B=cp.asarray(hB)
    C=cp.asarray(hC) if use_C else None
    D=cp.empty((M,N), np.float32)
    Z=cp.empty((M,N), np.float32) if save_preact else None
    bias = cp.asarray(hBias) if hBias is not None else None

    # --- Forward(EX)
    px = ge2.GemmBiasActParamsEx()
    px.M,px.N,px.K = M,N,K
    px.lda,px.ldb,px.ldc,px.ldd = K,N,N,N
    px.alpha,px.beta = float(alpha), float(beta)
    px.use_C = 1 if use_C else 0
    px.has_bias = 1 if (bias is not None) else 0
    px.bias_kind = bias_kind
    px.act_kind  = act_kind
    px.leaky_slope = float(leaky)
    px.save_preact = 1 if save_preact else 0
    px.ldZ = 0

    ge2.gemm_bias_act_f32_ex(
        int(A.data.ptr), int(B.data.ptr),
        int(C.data.ptr) if C is not None else None,
        int(D.data.ptr),
        int(bias.data.ptr) if bias is not None else None,
        int(Z.data.ptr) if Z is not None else None,
        px, None
    )

    # CPU pre
    hAB = hA @ hB
    hPre = alpha*hAB + (beta*hC if use_C else 0)
    if bias_kind == ge2.BiasKind.Scalar and hBias is not None:
        hPre = hPre + hBias[0]
    elif bias_kind == ge2.BiasKind.PerM and hBias is not None:
        hPre = hPre + hBias[:,None]
    elif bias_kind == ge2.BiasKind.PerN and hBias is not None:
        hPre = hPre + hBias[None,:]

    # FWD probe: check broadcast direction inference
    Dgpu = cp.asnumpy(D)
    # FWD probe: make a WRONG (but shape-safe) broadcast to catch axis mistakes
    if hBias is not None:
        if bias_kind == ge2.BiasKind.PerM:
            # 정답: + hBias[:, None]
            # 오답(PerN처럼): 길이를 N에 맞춰 변형한 뒤 열방향으로 더한다
            if N <= M:
                bN = hBias[:N]
            else:
                bN = np.pad(hBias, (0, N - M), mode='wrap')  # 또는 'edge'
            hPre_alt = alpha*hAB + (beta*hC if use_C else 0) + bN[None, :]
        elif bias_kind == ge2.BiasKind.PerN:
            # 정답: + hBias[None, :]
            # 오답(PerM처럼): 길이를 M에 맞춰 변형한 뒤 행방향으로 더한다
            if M <= N:
                bM = hBias[:M]
            else:
                bM = np.pad(hBias, (0, M - N), mode='wrap')
            hPre_alt = alpha*hAB + (beta*hC if use_C else 0) + bM[:, None]
        else:
            hPre_alt = hPre.copy()
    else:
        hPre_alt = hPre.copy()


    # Try several activations to see which matches GPU FWD best
    act_names = ["None","ReLU","LeakyReLU","GELU","Sigmoid","Tanh"]
    fwd_ranking=[]
    for name in act_names:
        y1 = act_fn(name)(hPre, leaky)
        y2 = act_fn(name)(hPre_alt, leaky)
        diff1 = max_abs_diff(Dgpu, y1)
        diff2 = max_abs_diff(Dgpu, y2)
        fwd_ranking.append((name, "pre", diff1))
        fwd_ranking.append((name, "alt", diff2))
    fwd_ranking.sort(key=lambda x: x[2])

    # --- Backward (make grads)
    hGY = rng.standard_normal((M,N), np.float32)
    gY = cp.asarray(hGY)
    gA = cp.empty((M,K), np.float32); gA.fill(0)
    gB = cp.empty((K,N), np.float32); gB.fill(0)
    gC = cp.empty((M,N), np.float32) if use_C else None
    if gC is not None: gC.fill(0)
    if bias_kind == ge2.BiasKind.Scalar: gBias = cp.empty((1,), np.float32)
    elif bias_kind == ge2.BiasKind.PerM: gBias = cp.empty((M,), np.float32)
    elif bias_kind == ge2.BiasKind.PerN: gBias = cp.empty((N,), np.float32)
    else: gBias=None
    if gBias is not None: gBias.fill(0)

    pb = ge2.GemmBiasActBwdParams()
    pb.M,pb.N,pb.K = M,N,K
    pb.lda,pb.ldb,pb.ldc = K,N,N
    pb.ldgY,pb.ldZ = N,N
    pb.ldgA,pb.ldgB = K,N
    pb.ldgC = N
    pb.alpha,pb.beta = float(alpha), float(beta)
    pb.bias_kind,pb.act_kind = bias_kind, act_kind
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

    dA = cp.asnumpy(gA); dB = cp.asnumpy(gB)
    dC = cp.asnumpy(gC) if gC is not None else None
    dBias = cp.asnumpy(gBias) if gBias is not None else None

    # BWD probe: which derivative did GPU effectively use?
    bwd_ranking=[]
    for name in act_names:
        GZ_pre = hGY * dact_fn(name)(hPre, leaky)
        GA = (GZ_pre @ hB.T)            # 수학적으로는 alpha*...가 정석. (GPU 구현과의 차이는 아래서 두 경우 모두 평가)
        GB = (hA.T @ GZ_pre)
        GAa = alpha*GA; GBa = alpha*GB
        diffs = []
        diffs.append(("no_alpha", name, max_abs_diff(dA, GA) + max_abs_diff(dB, GB)))
        diffs.append(("with_alpha", name, max_abs_diff(dA, GAa) + max_abs_diff(dB, GBa)))
        bwd_ranking.extend(diffs)
    bwd_ranking.sort(key=lambda x: x[2])

    # Bias grad check
    def bias_reduce(GZ):
        if bias_kind == ge2.BiasKind.Scalar: return np.array([np.sum(GZ, dtype=np.float32)], np.float32)
        if bias_kind == ge2.BiasKind.PerM:   return np.sum(GZ, axis=1, dtype=np.float32)
        if bias_kind == ge2.BiasKind.PerN:   return np.sum(GZ, axis=0, dtype=np.float32)
        return None

    bias_ranking=[]
    if dBias is not None:
        for name in act_names:
            GZ = hGY * dact_fn(name)(hPre, leaky)
            ref = bias_reduce(GZ)
            diff = max_abs_diff(dBias, ref)
            bias_ranking.append((name, diff))
        bias_ranking.sort(key=lambda x: x[1])

    # dC check (if any)
    dC_diff = None
    if dC is not None:
        GZ = hGY * dact_fn("GELU" if act_kind==ge2.ActKind.GELU else "ReLU" if act_kind==ge2.ActKind.ReLU else "LeakyReLU")(hPre, leaky)
        refC = beta*GZ
        dC_diff = max_abs_diff(dC, refC)

    # print summary
    print("="*70)
    ak = int(act_kind) if hasattr(act_kind,"__int__") or isinstance(act_kind,int) else act_kind
    bk = int(bias_kind) if hasattr(bias_kind,"__int__") or isinstance(bias_kind,int) else bias_kind
    print(f"[Probe] M={M},N={N},K={K}, act={act_kind}({ak}), bias={bias_kind}({bk}), use_C={use_C}")
    print("FWD best matches (activation, broadcast[pre/alt], max|diff|):")
    for name,mode,d in fwd_ranking[:6]:
        print(f"  - {name:10s} | {mode} | {d:.6f}")
    print("BWD best matches (alpha_mode, activation, sum(diffs dA+dB)):")
    for mode,name,d in bwd_ranking[:6]:
        print(f"  - {mode:9s} | {name:10s} | {d:.6f}")
    if bias_ranking:
        print("Bias grad best matches (activation, max|diff|):")
        for name,d in bias_ranking[:6]:
            print(f"  - {name:10s} | {d:.6f}")
    if dC_diff is not None:
        print(f"dC check (assuming spec): max|diff|={dC_diff:.6f}")
    print("="*70)

if __name__ == "__main__":
    # Case2 (문제 케이스)
    probe_case(M=64, N=64, K=32,
               act_kind=ge2.ActKind.ReLU,
               bias_kind=ge2.BiasKind.Scalar,
               use_C=False, save_preact=True,
               alpha=1.0, beta=0.0, leaky=0.02, seed=0)

    # Case3 (문제 케이스)
    probe_case(M=96, N=80, K=40,
               act_kind=ge2.ActKind.LeakyReLU,
               bias_kind=ge2.BiasKind.PerM,
               use_C=True, save_preact=True,
               alpha=1.1, beta=0.9, leaky=0.02, seed=0)
