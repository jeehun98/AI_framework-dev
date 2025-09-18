# ge2_check_backward_probe_plus.py
# - FWD/BWD 활성화(및 브로드캐스트) 경로 역추정
# - BWD에서 어떤 도함수(enum)가 적용됐는지 랭킹
# - LeakyReLU의 실제 사용된 기울기 s_hat 역추정 (PerM 케이스 중심)
# - PerM/PerN의 "고의로 틀린" 브로드캐스트도 shape-safe 하게 비교

import numpy as np
import cupy as cp
import graph_executor_v2 as ge2

# ---------- reference activations ----------
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

def act_fn(name):
    def _relu(x): return np.maximum(x, 0)
    def _leaky(x,a): return np.where(x>0, x, a*x)
    def _sigmoid(x): return 1/(1+np.exp(-x))
    def _tanh(x): return np.tanh(x)
    if name == "None":      return lambda x, a=0.01: x
    if name == "ReLU":      return lambda x, a=0.01: _relu(x)
    if name == "LeakyReLU": return lambda x, a=0.01: _leaky(x, a)
    if name == "GELU":      return lambda x, a=0.01: gelu(x)
    if name == "Sigmoid":   return lambda x, a=0.01: _sigmoid(x)
    if name == "Tanh":      return lambda x, a=0.01: _tanh(x)
    raise ValueError(name)

def dact_fn(name):
    def _drelu(x,a): return (x>0).astype(x.dtype)
    def _dleaky(x,a): return np.where(x>0, 1, a).astype(x.dtype)
    def _dsigmoid(x,a):
        s = 1/(1+np.exp(-x)); return s*(1-s)
    def _dtanh(x,a):
        t=np.tanh(x); return 1-t*t
    if name == "None":      return lambda x,a: np.ones_like(x)
    if name == "ReLU":      return lambda x,a: _drelu(x,a)
    if name == "LeakyReLU": return lambda x,a: _dleaky(x,a)
    if name == "GELU":      return lambda x,a: dgelu(x)
    if name == "Sigmoid":   return lambda x,a: _dsigmoid(x,a)
    if name == "Tanh":      return lambda x,a: _dtanh(x,a)
    raise ValueError(name)

def max_abs_diff(a,b): 
    return float(np.max(np.abs(a-b)))

def shape_safe_wrong_broadcast(hPre_base, hBias, bias_kind, M, N, alpha, beta, hAB, hC, use_C):
    """
    정답 브로드캐스트와 '고의로 틀린' 브로드캐스트를 모두 shape-safe하게 생성.
    반환: (hPre_correct, hPre_alt_wrong)
    """
    hPre_correct = hPre_base
    if hBias is None:
        return hPre_correct, hPre_correct.copy()

    if bias_kind == ge2.BiasKind.PerM:
        # 정답: + hBias[:, None]
        hPre_correct = alpha*hAB + (beta*hC if use_C else 0) + hBias[:, None]
        # 틀린 버전(PerN처럼 열방향에 더함) → 길이를 N에 맞추어 패딩/절삭
        if N <= hBias.shape[0]:
            bN = hBias[:N]
        else:
            bN = np.pad(hBias, (0, N - hBias.shape[0]), mode='wrap')
        hPre_alt = alpha*hAB + (beta*hC if use_C else 0) + bN[None, :]

    elif bias_kind == ge2.BiasKind.PerN:
        # 정답: + hBias[None, :]
        hPre_correct = alpha*hAB + (beta*hC if use_C else 0) + hBias[None, :]
        # 틀린 버전(PerM처럼 행방향에 더함) → 길이를 M에 맞춤
        if M <= hBias.shape[0]:
            bM = hBias[:M]
        else:
            bM = np.pad(hBias, (0, M - hBias.shape[0]), mode='wrap')
        hPre_alt = alpha*hAB + (beta*hC if use_C else 0) + bM[:, None]

    elif bias_kind == ge2.BiasKind.Scalar:
        # 정답: + hBias[0]
        hPre_correct = alpha*hAB + (beta*hC if use_C else 0) + hBias[0]
        # 틀린 버전: 사실상 동일(스칼라는 대칭), 그래도 분기 통일
        hPre_alt = hPre_correct.copy()
    else:
        hPre_alt = hPre_correct.copy()

    return hPre_correct, hPre_alt

def infer_leaky_slope_from_output(Dgpu, pre):
    """
    음수 영역에서 y ≈ s * pre 라고 보고 최소자승으로 s 추정.
    pre<0인 영역만 사용. 유효한 점이 없으면 None 반환.
    """
    pre_neg = pre < 0
    cnt = int(np.sum(pre_neg))
    if cnt == 0:
        return None
    Dy = cp.asnumpy(Dgpu)
    num = float(np.sum(Dy[pre_neg] * pre[pre_neg]))
    den = float(np.sum(pre[pre_neg]*pre[pre_neg]))
    if den == 0.0:
        return None
    return num/den

def probe_case(M, N, K,
               act_kind, bias_kind,
               use_C, save_preact,
               alpha, beta, leaky, seed=0):
    rng = np.random.default_rng(seed)
    hA = rng.standard_normal((M,K), np.float32)
    hB = rng.standard_normal((K,N), np.float32)
    hC = rng.standard_normal((M,N), np.float32)
    # host bias
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

    # CPU pre (정답 수식)
    hAB = hA @ hB
    hPre_base = alpha*hAB + (beta*hC if use_C else 0)
    if hBias is not None:
        if bias_kind == ge2.BiasKind.Scalar:
            hPre = hPre_base + hBias[0]
        elif bias_kind == ge2.BiasKind.PerM:
            hPre = hPre_base + hBias[:, None]
        elif bias_kind == ge2.BiasKind.PerN:
            hPre = hPre_base + hBias[None, :]
        else:
            hPre = hPre_base
    else:
        hPre = hPre_base

    # 올바른/틀린 브로드캐스트 버전을 둘 다 만들어 비교
    hPre_corr, hPre_alt = shape_safe_wrong_broadcast(hPre_base, hBias, bias_kind, M, N, alpha, beta, hAB, hC, use_C)

    Dgpu = cp.asnumpy(D)

    # --- FWD ranking: 어떤 activation + broadcast가 GPU FWD와 제일 비슷한가?
    act_names = ["None","ReLU","LeakyReLU","GELU","Sigmoid","Tanh"]
    fwd_ranking=[]
    for name in act_names:
        y1 = act_fn(name)(hPre_corr, leaky)
        y2 = act_fn(name)(hPre_alt, leaky)
        diff1 = max_abs_diff(Dgpu, y1)
        diff2 = max_abs_diff(Dgpu, y2)
        fwd_ranking.append((name, "pre", diff1))
        fwd_ranking.append((name, "alt", diff2))
    fwd_ranking.sort(key=lambda x: x[2])

    # --- BWD 실행 (GPU)
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

    # --- BWD ranking: 어떤 도함수/alpha 적용 케이스가 dA+dB를 가장 잘 설명?
    bwd_ranking=[]
    for name in act_names:
        GZ = hGY * dact_fn(name)(hPre, leaky)  # spec상 pre-activation 사용
        GA = (GZ @ hB.T);   GB = (hA.T @ GZ)
        GAa = alpha*GA;     GBa = alpha*GB
        bwd_ranking.append(("no_alpha",  name, max_abs_diff(dA, GA)  + max_abs_diff(dB, GB)))
        bwd_ranking.append(("with_alpha",name, max_abs_diff(dA, GAa) + max_abs_diff(dB, GBa)))
    bwd_ranking.sort(key=lambda x: x[2])

    # --- Bias grad ranking
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

    # --- dC check
    dC_diff = None
    if dC is not None:
        GZ = hGY * dact_fn("GELU" if act_kind==ge2.ActKind.GELU else
                           "ReLU" if act_kind==ge2.ActKind.ReLU else
                           "LeakyReLU" if act_kind==ge2.ActKind.LeakyReLU else
                           "Sigmoid" if act_kind==ge2.ActKind.Sigmoid else
                           "Tanh" if act_kind==ge2.ActKind.Tanh else "None")(hPre, leaky)
        refC = beta*GZ
        dC_diff = max_abs_diff(dC, refC)

    # --- Leaky slope 역추정 (FWD 기준, PerM/Leaky에서 특히 유용)
    s_hat = None
    if act_kind == ge2.ActKind.LeakyReLU:
        # spec상 올바른 pre 사용
        s_hat = infer_leaky_slope_from_output(D, hPre)

    # --- 출력
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
    if s_hat is not None:
        print(f"[Infer] leaky_slope_hat ≈ {s_hat:.6f} (requested={leaky})")
    print("="*70)

if __name__ == "__main__":
    # 문제 케이스 1: ReLU + Scalar + no C
    probe_case(M=64, N=64, K=32,
               act_kind=ge2.ActKind.ReLU,
               bias_kind=ge2.BiasKind.Scalar,
               use_C=False, save_preact=True,
               alpha=1.0, beta=0.0, leaky=0.02, seed=0)

    # 문제 케이스 2: LeakyReLU + PerM + use C
    probe_case(M=96, N=80, K=40,
               act_kind=ge2.ActKind.LeakyReLU,
               bias_kind=ge2.BiasKind.PerM,
               use_C=True, save_preact=True,
               alpha=1.1, beta=0.9, leaky=0.02, seed=0)
