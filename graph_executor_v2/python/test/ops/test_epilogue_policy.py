# python/test/ops/test_epilogue_policy.py
import os, sys, ctypes, math
from pathlib import Path
import numpy as np

# ================================
# 0) 경로 설정 & CUDA DLL 경로 선등록 (Windows)
# ================================
THIS = Path(__file__).resolve().parent
ROOT = (THIS / ".." / ".." / "..").resolve()
PKG_OPS = ROOT / "python" / "graph_executor_v2" / "ops"

def _prime_cuda_dll_dirs_windows():
    """
    Import 전에 OS 로더가 cudart64_*.dll을 찾을 수 있도록
    CUDA의 bin 경로들을 add_dll_directory로 '선등록'한다.
    """
    added = set()

    def _add_dir(p: Path):
        try:
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(p))
                added.add(str(p))
        except Exception:
            pass

    # 1) PATH 스캔: cudart64_*.dll이 들어있는 폴더 추가
    for p in os.environ.get("PATH", "").split(os.pathsep):
        p = p.strip('"')
        if not p:
            continue
        try:
            pp = Path(p)
            if list(pp.glob("cudart64_*.dll")):
                _add_dir(pp)
        except Exception:
            pass

    # 2) CUDA_PATH / CUDA_PATH_V12_* 에 지정된 bin 추가
    for k, v in os.environ.items():
        if k.startswith("CUDA_PATH") and v:
            binp = Path(v) / "bin"
            if binp.is_dir():
                _add_dir(binp)

    # 3) 기본 설치 루트에서 최신 버전의 bin 추가
    default_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if default_root.exists():
        for sub in sorted(default_root.glob("v*"), reverse=True):
            binp = sub / "bin"
            if binp.is_dir():
                _add_dir(binp)

    return sorted(added)

if os.name == "nt":
    _ = _prime_cuda_dll_dirs_windows()

# ================================
# 1) pybind 모듈 import
#    (epilogue_pybind.cpp: PYBIND11_MODULE(_ops_epilogue, m) 이어야 함)
# ================================
if str(PKG_OPS) not in sys.path:
    sys.path.insert(0, str(PKG_OPS))
import _ops_epilogue as ep

# ================================
# 2) cudart 동적 로드 (테스트에서 cudaMalloc 등 직접 호출 용)
# ================================
def _load_cudart():
    if os.name != "nt":
        return ctypes.CDLL("libcudart.so")

    # Windows: 실제 파일 경로를 찾아서 로드
    candidates = []

    # a) PATH
    for p in os.environ.get("PATH", "").split(os.pathsep):
        p = p.strip('"')
        if not p:
            continue
        try:
            for dll in Path(p).glob("cudart64_*.dll"):
                candidates.append(dll.resolve())
        except Exception:
            pass

    # b) CUDA_PATH* \bin
    for k, v in os.environ.items():
        if k.startswith("CUDA_PATH") and v:
            binp = Path(v) / "bin"
            if binp.is_dir():
                for dll in binp.glob("cudart64_*.dll"):
                    candidates.append(dll.resolve())

    # c) 기본 설치 루트
    default_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if default_root.exists():
        for sub in sorted(default_root.glob("v*"), reverse=True):
            binp = sub / "bin"
            if binp.is_dir():
                for dll in binp.glob("cudart64_*.dll"):
                    candidates.append(dll.resolve())

    # 중복 제거 후 최신 경로 우선
    candidates = sorted(set(map(str, candidates)), reverse=True)
    if not candidates:
        raise FileNotFoundError(
            "Could not find cudart64_*.dll. "
            "Add CUDA\\vXX.X\\bin to PATH or set CUDA_PATH."
        )

    dll_path = candidates[0]
    # 디렉토리 등록(보조)
    try:
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(Path(dll_path).parent))
    except Exception:
        pass
    return ctypes.CDLL(dll_path)

cudart = _load_cudart()
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2
H2D, D2H = cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost

# ================================
# 3) CUDA 런타임 래퍼
# ================================
def cuda_check(code, msg):
    if code != 0:
        raise RuntimeError(f"[CUDA] {msg} -> err={code}")

def cudaMalloc(nbytes):
    ptr = ctypes.c_void_p()
    cuda_check(cudart.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
    return ptr

def cudaMemcpy(dst, src, nbytes, kind):
    cuda_check(cudart.cudaMemcpy(dst, src, nbytes, kind), "cudaMemcpy")

def cudaDeviceSynchronize():
    cuda_check(cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")

# ================================
# 4) 테스트 유틸
# ================================
def run_case(M=512, N=512, act=ep.ActKind.GELU, p=0.5, save_mask=True, with_bias=True, seed=1234):
    MN = M * N
    hx = (np.arange(MN, dtype=np.float32) % 97 * 0.03125 - 1.5).astype(np.float32)
    hb = np.full((N,), 0.25, dtype=np.float32)
    hy = np.empty((MN,), dtype=np.float32)
    hmask = np.empty((MN,), dtype=np.uint8)

    dx = cudaMalloc(hx.nbytes)
    dy = cudaMalloc(hy.nbytes)
    db = cudaMalloc(hb.nbytes) if with_bias else ctypes.c_void_p()
    dm = cudaMalloc(hmask.nbytes) if save_mask else ctypes.c_void_p()

    cudaMemcpy(dx, hx.ctypes.data_as(ctypes.c_void_p), hx.nbytes, H2D)
    if with_bias:
        cudaMemcpy(db, hb.ctypes.data_as(ctypes.c_void_p), hb.nbytes, H2D)

    pplan = ep.Plan()
    pplan.rows = M
    pplan.cols = N
    pplan.ld_x = N
    pplan.ld_y = N
    pplan.ld_bias = N
    pplan.attrs = ep.Attrs()
    pplan.attrs.act = act
    pplan.attrs.dropout_p = float(p)
    pplan.attrs.save_mask = bool(save_mask)
    pplan.attrs.seed = int(seed)

    tens = ep.Tensors()
    tens.x = dx.value
    tens.y = dy.value
    tens.bias = (db.value if with_bias else 0)
    tens.mask_out = (dm.value if save_mask else 0)

    st = ep.run(pplan, tens, ep.DType.F32)
    assert st == 0, f"ep.run failed: {st}"
    cudaDeviceSynchronize()

    cudaMemcpy(ctypes.c_void_p(hy.ctypes.data), dy, hy.nbytes, D2H)
    if save_mask:
        cudaMemcpy(ctypes.c_void_p(hmask.ctypes.data), dm, hmask.nbytes, D2H)
    else:
        hmask = None

    return hy, hmask

def assert_prob_close(ones, total, keep, z=5.0):
    n = float(total)
    mu = n * keep
    sigma = math.sqrt(max(mu * (1.0 - keep), 1e-12))
    diff = abs(float(ones) - mu)
    assert diff <= z * sigma, f"mask ones={ones}/{total}, keep={keep:.3f}, |diff|={diff:.1f} > {z}σ={z*sigma:.1f}"

# ================================
# 5) 본 테스트
# ================================
def test_epilogue_dropout_like():
    M, N = 512, 512
    MN = M * N

    # 기준: p=0 (dropout off)
    y_ref, _ = run_case(M, N, act=ep.ActKind.GELU, p=0.0, save_mask=True, with_bias=True, seed=777)
    mean_ref = float(y_ref.mean())

    # p=0.5, save_mask=True
    y1, m1 = run_case(M, N, act=ep.ActKind.GELU, p=0.5, save_mask=True, with_bias=True, seed=777)
    assert m1 is not None
    ones = int(m1.sum())
    keep = 0.5
    assert_prob_close(ones, MN, keep, z=5.0)

    # 스케일 검증: inverted dropout이면 평균 유지
    mean_y1 = float(np.mean(y1))
    assert np.isclose(mean_y1, mean_ref, rtol=5e-2, atol=5e-3), f"mean mismatch: {mean_y1} vs {mean_ref}"

    # 결정성: 같은 seed 두 번 → 동일
    y2, m2 = run_case(M, N, act=ep.ActKind.GELU, p=0.5, save_mask=True, with_bias=True, seed=777)
    assert np.array_equal(m1, m2), "determinism(mask)"
    assert np.array_equal(y1, y2), "determinism(y)"

    # seed 바꾸면 보통 달라야 함
    y3, m3 = run_case(M, N, act=ep.ActKind.GELU, p=0.5, save_mask=True, with_bias=True, seed=778)
    if np.array_equal(m1, m3):
        print("[WARN] different seed produced same mask (rare)")

    # p=0 경계: 전부 1이어야 함
    y0, m0 = run_case(256, 256, act=ep.ActKind.ReLU, p=0.0, save_mask=True, with_bias=False, seed=42)
    assert int(m0.sum()) == m0.size

    # p=0.9 희박: keep≈0.1
    yh, mh = run_case(512, 256, act=ep.ActKind.ReLU, p=0.9, save_mask=True, with_bias=True, seed=123)
    assert_prob_close(int(mh.sum()), yh.size, 0.1, z=5.0)

    # save_mask=False 경로도 런치만 확인
    yx, mx = run_case(384, 768, act=ep.ActKind.GELU, p=0.3, save_mask=False, with_bias=True, seed=9999)
    assert mx is None

# 직접 실행 시
if __name__ == "__main__":
    test_epilogue_dropout_like()
    print("[OK] epilogue dropout-like test passed.")
