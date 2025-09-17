# ge2_smoke_test.py
# 단일 파일 스모크 테스트: DLL 경로 세팅 → 모듈 import → GEMM+Bias+Act(EX) 검증
# 사용: python ge2_smoke_test.py

import os, sys, ctypes, importlib
from dataclasses import dataclass
import numpy as np

# ---------------------------
# 0) DLL 경로 추가 (환경에 맞게 수정 가능)
# ---------------------------
def _safely_add_dll_dir(p):
    if p and os.path.isdir(p):
        try:
            os.add_dll_directory(p)  # Python 3.8+
            print(f"[dll] added: {p}")
        except Exception as e:
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
            print(f"[dll] PATH prepend: {p} ({e})")

HERE = os.path.dirname(os.path.abspath(__file__))

# (필요에 따라 경로 추가: pyd 위치, 빌드/런타임 dll 위치, CUDA bin)
# 1) pyd와 같은 폴더
_safely_add_dll_dir(HERE)

# 2) 빌드 산출물 후보 (프로젝트 구조에 맞춰 수정)
for c in [
    os.path.join(HERE),                     # 동일 폴더
    os.path.join(HERE, "bin"),
    os.path.join(HERE, "build"),
    os.path.join(HERE, "build", "bin"),
    os.path.join(HERE, "..", "build"),
    os.path.join(HERE, "..", "build", "bin"),
]:
    _safely_add_dll_dir(os.path.abspath(c))

# 3) CUDA 경로
cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
if cuda_path:
    _safely_add_dll_dir(os.path.join(cuda_path, "bin"))
# 설치 버전에 맞게 하드코딩(예: CUDA 12.6)
_safely_add_dll_dir(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")

# ---------------------------
# 1) 모듈 import 전 저수준 로드 진단(누락 DLL 이름 추적에 유용)
# ---------------------------
pyd_path = os.path.join(HERE, "graph_executor_v2.pyd")
if os.path.isfile(pyd_path):
    try:
        ctypes.WinDLL(pyd_path)
        print("[ctypes] low-level load OK")
    except Exception as e:
        print("[ctypes] low-level load FAIL:", e)

# ---------------------------
# 2) 모듈 import
# ---------------------------
try:
    ge2 = importlib.import_module("graph_executor_v2")
    print("[import] graph_executor_v2 OK")
except Exception as e:
    print("[import] FAIL:", repr(e))
    sys.exit(1)

# 선택 의존: CuPy / Torch (둘 중 하나만 있으면 그걸로 테스트)
try:
    import cupy as cp
    HAS_CUPY = True
    print("[env] CuPy detected")
except Exception:
    HAS_CUPY = False

try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
    if HAS_TORCH:
        print("[env] PyTorch CUDA detected")
except Exception:
    HAS_TORCH = False

if not (HAS_CUPY or HAS_TORCH):
    print("[env] No CuPy/PyTorch CUDA available. Install one to run the test.")
    sys.exit(2)

# ---------------------------
# 3) 도우미: 파라미터/실행/레퍼런스
# ---------------------------
@dataclass
class ParamsExCfg:
    M: int; N: int; K: int
    lda: int; ldb: int; ldc: int; ldd: int
    alpha: float = 1.0; beta: float = 0.0
    use_C: int = 0; has_bias: int = 0
    bias_kind: str = "PerN"      # "Scalar"|"PerM"|"PerN"
    act_kind: str = "GELU"       # "None"|"ReLU"|"LeakyReLU"|"GELU"|"Sigmoid"|"Tanh"
    leaky_slope: float = 0.01

def make_params_ex(cfg: ParamsExCfg):
    px = ge2.GemmBiasActParamsEx()
    px.M, px.N, px.K = cfg.M, cfg.N, cfg.K
    px.lda, px.ldb, px.ldc, px.ldd = cfg.lda, cfg.ldb, cfg.ldc, cfg.ldd
    px.alpha, px.beta = float(cfg.alpha), float(cfg.beta)
    px.use_C, px.has_bias = int(cfg.use_C), int(cfg.has_bias)
    px.leaky_slope = float(cfg.leaky_slope)

    # enum 매핑: "None"은 예약어라 속성 접근 불가 → dict-style로 접근
    px.bias_kind = {
        "Scalar": ge2.BiasKind.Scalar,
        "PerM":   ge2.BiasKind.PerM,
        "PerN":   ge2.BiasKind.PerN,
    }[cfg.bias_kind]

    px.act_kind = {
        "None":      getattr(ge2.ActKind, "None"),
        "ReLU":      ge2.ActKind.ReLU,
        "LeakyReLU": ge2.ActKind.LeakyReLU,
        "GELU":      ge2.ActKind.GELU,
        "Sigmoid":   ge2.ActKind.Sigmoid,
        "Tanh":      ge2.ActKind.Tanh,
    }[cfg.act_kind]
    return px

def act_forward(x: np.ndarray, kind: str, leaky_slope: float = 0.01):
    if kind == "None":
        return x
    if kind == "ReLU":
        return np.maximum(x, 0.0, dtype=x.dtype)
    if kind == "LeakyReLU":
        return np.where(x > 0, x, leaky_slope * x, dtype=x.dtype)
    if kind == "GELU":
        # tanh approx (Hendrycks & Gimpel)
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * np.power(x, 3))))
    if kind == "Sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if kind == "Tanh":
        return np.tanh(x)
    raise ValueError(f"Unknown act kind: {kind}")

def add_bias(Z: np.ndarray, bias: np.ndarray | float | None, kind: str):
    if bias is None:
        return Z
    if kind == "Scalar":
        return Z + float(bias)
    if kind == "PerM":
        b = np.asarray(bias, dtype=Z.dtype).reshape((-1, 1))
        return Z + b
    if kind == "PerN":
        b = np.asarray(bias, dtype=Z.dtype).reshape((1, -1))
        return Z + b
    raise ValueError(f"Unknown bias kind: {kind}")

def reference_np(hA, hB, hC_or_None, bias_or_None, cfg: ParamsExCfg):
    Z = cfg.alpha * (hA @ hB)
    if cfg.use_C and hC_or_None is not None:
        Z = Z + cfg.beta * hC_or_None
    if cfg.has_bias:
        Z = add_bias(Z, bias_or_None, cfg.bias_kind)
    Z = act_forward(Z, cfg.act_kind, cfg.leaky_slope)
    return Z

def max_abs_diff(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.max(np.abs(x - y)))

def _ptr_cupy(x):  # CuPy
    return int(x.data.ptr)

def _stream_cupy():
    import cupy as cp
    return int(cp.cuda.get_current_stream().ptr)

def _ptr_torch(x):  # Torch
    return int(x.data_ptr())

def _stream_torch(dev):
    import torch
    return int(torch.cuda.current_stream(device=dev).cuda_stream)

# ---------------------------
# 4) 실행: CuPy 우선, 없으면 Torch
# ---------------------------
def run_cupy():
    import cupy as cp

    M, N, K = 128, 128, 128
    rng = np.random.default_rng(0)
    hA = rng.standard_normal((M, K), np.float32)
    hB = rng.standard_normal((K, N), np.float32)
    hC = rng.standard_normal((M, N), np.float32)
    hBiasN = np.full((N,), 0.1, np.float32)

    A = cp.asarray(hA); B = cp.asarray(hB); C = cp.asarray(hC)
    D = cp.empty((M, N), np.float32); biasN = cp.asarray(hBiasN)

    cfg = ParamsExCfg(
        M=M, N=N, K=K,
        lda=K, ldb=N, ldc=N, ldd=N,        # row-major
        alpha=1.0, beta=1.0,
        use_C=1, has_bias=1,
        bias_kind="PerN",
        act_kind="GELU",
        leaky_slope=0.01,
    )
    px = make_params_ex(cfg)

    a, b, c, d, bias = _ptr_cupy(A), _ptr_cupy(B), _ptr_cupy(C), _ptr_cupy(D), _ptr_cupy(biasN)
    stream = _stream_cupy()

    # 호출: C의 시그니처에 맞춰 None 처리
    ge2.gemm_bias_act_f32_ex(a, b, c if px.use_C else None,
                             d, bias if px.has_bias else None,
                             px, stream)

    # Ref & 비교
    hRef = reference_np(hA, hB, hC, hBiasN, cfg)
    hD = cp.asnumpy(D)   # 동기화 포함
    diff = max_abs_diff(hD, hRef)
    print(f"[CuPy] max|diff| = {diff}")
    return diff

def run_torch():
    import torch
    dev = torch.device("cuda")
    M, N, K = 128, 128, 128
    rng = np.random.default_rng(0)
    hA = rng.standard_normal((M, K), np.float32)
    hB = rng.standard_normal((K, N), np.float32)
    hC = rng.standard_normal((M, N), np.float32)
    hBiasN = np.full((N,), 0.1, np.float32)

    A = torch.from_numpy(hA).to(dev)
    B = torch.from_numpy(hB).to(dev)
    C = torch.from_numpy(hC).to(dev)
    D = torch.empty((M, N), device=dev, dtype=torch.float32)
    biasN = torch.from_numpy(hBiasN).to(dev)

    cfg = ParamsExCfg(
        M=M, N=N, K=K,
        lda=K, ldb=N, ldc=N, ldd=N,
        alpha=1.0, beta=1.0,
        use_C=1, has_bias=1,
        bias_kind="PerN",
        act_kind="GELU",
        leaky_slope=0.01,
    )
    px = make_params_ex(cfg)

    a, b, c, d, bias = _ptr_torch(A), _ptr_torch(B), _ptr_torch(C), _ptr_torch(D), _ptr_torch(biasN)
    stream = _stream_torch(dev)

    ge2.gemm_bias_act_f32_ex(a, b, c if px.use_C else None,
                             d, bias if px.has_bias else None,
                             px, stream)

    # Ref & 비교
    torch.cuda.synchronize(dev)
    hRef = reference_np(hA, hB, hC, hBiasN, cfg)
    hD = D.cpu().numpy()
    diff = max_abs_diff(hD, hRef)
    print(f"[Torch] max|diff| = {diff}")
    return diff

# ---------------------------
# 5) 메인
# ---------------------------
if __name__ == "__main__":
    try:
        if HAS_CUPY:
            diff = run_cupy()
        else:
            diff = run_torch()
        atol = 3e-5
        ok = diff <= atol
        print(f"[RESULT] OK={ok} (atol={atol})")
        sys.exit(0 if ok else 3)
    except Exception as e:
        print("[ERROR]", repr(e))
        sys.exit(4)
