import os
import sys

def find_and_add_project_root(target_dirname="dev"):
    current_path = os.path.abspath(__file__)
    while True:
        current_path = os.path.dirname(current_path)
        if os.path.basename(current_path) == target_dirname:
            if current_path not in sys.path:
                sys.path.insert(0, current_path)
            return current_path
        if current_path == os.path.dirname(current_path):
            raise RuntimeError(f"'{target_dirname}' 디렉토리를 찾을 수 없습니다.")

def setup_paths():
    # 기존 루트 등록
    project_root = find_and_add_project_root("dev")

    # CUDA 연산 모듈 등록
    matrix_ops_path = os.path.join(
        project_root, "backend", "operaters", "build", "lib.win-amd64-cpython-312"
    )
    if os.path.exists(matrix_ops_path):
        sys.path.append(matrix_ops_path)

    # ✅ [추가] activations 모듈 경로 등록
    activations_path = os.path.join(
        project_root, "backend", "activations", "build", "lib.win-amd64-cpython-312"
    )
    if os.path.exists(activations_path):
        sys.path.append(activations_path)

    # CUDA DLL 등록
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
    if os.path.exists(cuda_path):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(cuda_path)
        else:
            os.environ["PATH"] = cuda_path + os.pathsep + os.environ["PATH"]
