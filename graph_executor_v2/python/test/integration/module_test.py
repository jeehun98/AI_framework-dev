# test_import_ops_gemm.py
import sys, os, importlib, ctypes, glob, platform

print("=== PYTHON ===")
print(sys.version, platform.architecture())
print("executable:", sys.executable)

def add_path(p):
    if p and os.path.isdir(p):
        try:
            os.add_dll_directory(p)
            print(f"[PATH+] {p}")
        except Exception as e:
            print(f"[PATH!] {p} (skip: {e})")

# --- repo root on sys.path ---
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
print("\n=== REPO ===")
print("THIS:", THIS)
print("ROOT:", ROOT)

# --- locate ops dir & pyd ---
import graph_executor_v2 as ge
import graph_executor_v2.ops as ops
OPS_DIR = os.path.dirname(ops.__file__)
pyds = glob.glob(os.path.join(OPS_DIR, "_ops_gemm*.pyd"))
print("\n=== OPS ===")
print("ops dir:", OPS_DIR)
print("pyds:", pyds)

# --- candidate DLL dirs (priority order) ---
print("\n=== DLL DIRS (add_dll_directory) ===")
# 1) ops dir (pyd 옆 커스텀 DLL이 있다면 로더가 바로 찾게)
add_path(OPS_DIR)

# 2) CUDA PATH/bin
cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
if cuda_path:
    add_path(os.path.join(cuda_path, "bin"))
else:
    print("[WARN] CUDA_PATH not set")

# 3) CUDA 12.6 bin (명시)
add_path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")

# 4) NVTX 기본 설치 경로(있으면 추가)
add_path(r"C:\Program Files\NVIDIA Corporation\NVToolsExt\bin\x64")
add_path(r"C:\Program Files\NVIDIA Corporation\NVToolsExt\bin")  # 일부 설치본

# 5) MSVC 재배포 디렉토리 힌트(없어도 됨)
add_path(os.environ.get("VCToolsRedistDir", ""))

# --- pre-load dependencies to reveal the true missing one ---
print("\n=== PRELOAD CHECK ===")
dll_candidates = [
    # CUDA core
    "cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll",
    # NVTX
    "nvToolsExt64_1.dll",
    # MSVC / OpenMP
    "VCRUNTIME140.dll", "vcruntime140_1.dll", "MSVCP140.dll", "vcomp140.dll",
    # Python
    "python312.dll",
]
missing = []
for dll in dll_candidates:
    try:
        ctypes.WinDLL(dll)
        print("OK :", dll)
    except OSError as e:
        print("MISS:", dll, "-", e)
        missing.append((dll, str(e)))

# --- try import, stage 1 ---
print("\n=== IMPORT _ops_gemm (stage 1) ===")
try:
    mod = importlib.import_module("graph_executor_v2.ops._ops_gemm")
    print("Loaded:", mod)
    sys.exit(0)
except Exception as e:
    print("FAILED stage 1:", repr(e))

# --- extra diagnostics: show PATH head & sys.path head ---
print("\n=== ENV PATH (head) ===")
paths = os.environ.get("PATH", "").split(os.pathsep)
for i, p in enumerate(paths[:12]):
    print(f"{i:02d} {p}")
print("...")

print("\n=== sys.path (head) ===")
for i, p in enumerate(sys.path[:12]):
    print(f"{i:02d} {p}")
print("...")

# --- final hint if missing NVTX or others ---
if any(dll == "nvToolsExt64_1.dll" for dll, _ in missing):
    print("\n[HINT] nvToolsExt64_1.dll 이 누락되어 있으면 USE_NVTX=OFF 로 빌드하거나 "
          "CUDA\\bin 또는 NVToolsExt\\bin\\x64 경로를 add_dll_directory로 추가하세요.")

if any(dll == "vcruntime140_1.dll" for dll, _ in missing):
    print("[HINT] vcruntime140_1.dll 누락: 최신 MSVC 재배포(x64) 설치 필요.")

if any(dll == "vcomp140.dll" for dll, _ in missing):
    print("[HINT] vcomp140.dll 누락: /openmp 사용 시 필요. MSVC OpenMP 런타임 설치 또는 링크 옵션 확인.")

# --- try import, stage 2: explicit re-add and retry ---
print("\n=== IMPORT _ops_gemm (stage 2, retry) ===")
# 다시 한 번 ops 디렉토리 재추가(보수적)
add_path(OPS_DIR)
try:
    mod = importlib.import_module("graph_executor_v2.ops._ops_gemm")
    print("Loaded on retry:", mod)
    sys.exit(0)
except Exception as e:
    print("FAILED stage 2:", repr(e))
    print("\n[FINAL] 여전히 실패하면 _ops_gemm.pyd를 Dependencies.exe로 열어 "
          "빨간 항목(DLL) 이름을 확인해줘. 그 이름 그대로 알려주면 바로 패치 포인트 짚어줄게.")
