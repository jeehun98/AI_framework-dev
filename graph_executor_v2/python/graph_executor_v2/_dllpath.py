import os, sys, ctypes, pathlib

def _add(p: str):
    if p and os.path.isdir(p):
        try:
            os.add_dll_directory(p)
        except Exception:
            pass

def ensure_runtime_paths():
    # 1) ops 폴더 (커스텀 DLL 동반 배포시)
    here = pathlib.Path(__file__).resolve().parent
    ops_dir = (here / "ops").as_posix()
    _add(ops_dir)

    # 2) CUDA
    cuda = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda:
        _add(os.path.join(cuda, "bin"))

    # 3) 기본 설치 경로 힌트 (Windows)
    _add(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
    _add(r"C:\Program Files\NVIDIA Corporation\NVToolsExt\bin\x64")
    _add(r"C:\Program Files\NVIDIA Corporation\NVToolsExt\bin")

    # 4) (선택) 빠르게 누락 진단
    for dll in ("cudart64_12.dll","cublas64_12.dll","cublasLt64_12.dll",
                "nvToolsExt64_1.dll","vcruntime140_1.dll","MSVCP140.dll"):
        try:
            ctypes.WinDLL(dll)
        except OSError:
            # 조용히 통과 — 실제 임포트에서 실패하면 메시지로 안내
            pass
