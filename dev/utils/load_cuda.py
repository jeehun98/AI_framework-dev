# utils/load_cuda.py
import os
import sys
from typing import List, Optional

_DLL_DIR_HANDLES: List[object] = []

def ensure_cuda_dlls(cuda_path: Optional[str] = None) -> None:
    """
    Windows에서 CUDA DLL 탐색 경로를 보강한다.
    - Python 3.8+ : os.add_dll_directory 사용 (핸들을 전역에 보관해야 함)
    - cuda_path 미지정 시 환경변수 CUDA_PATH 사용
    """
    if sys.platform != "win32":
        return  # 비윈도우는 불필요

    # 이미 세팅했다면 중복 방지
    if _DLL_DIR_HANDLES:
        return

    cuda_path = cuda_path or os.environ.get("CUDA_PATH")
    if not cuda_path:
        raise RuntimeError("CUDA_PATH 환경변수가 설정되어 있지 않습니다.")

    cuda_bin = os.path.join(cuda_path, "bin")
    if not os.path.isdir(cuda_bin):
        raise RuntimeError(f"CUDA bin 디렉터리 없음: {cuda_bin}")

    # ※ 중요: 핸들을 전역 리스트에 보관해야 효력이 유지됨.
    handle = os.add_dll_directory(cuda_bin)
    _DLL_DIR_HANDLES.append(handle)

    # 필요시 추가 라이브러리 폴더도 등록 (예: cuDNN 등)
    # cudnn_bin = r"C:\tools\cudnn\bin"
    # if os.path.isdir(cudnn_bin):
    #     _DLL_DIR_HANDLES.append(os.add_dll_directory(cudnn_bin))
