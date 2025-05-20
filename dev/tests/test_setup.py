import os
import sys

def ensure_project_root_in_sys_path(target_dir="AI_framework-dev"):
    """
    AI_framework-dev 루트를 찾아 sys.path 에 추가합니다.
    """
    current = os.path.abspath(__file__)
    while True:
        current = os.path.dirname(current)
        if os.path.basename(current) == target_dir:
            if current not in sys.path:
                sys.path.insert(0, current)
            return current
        if current == os.path.dirname(current):
            raise RuntimeError(f"프로젝트 루트 '{target_dir}'를 찾을 수 없습니다.")

def find_and_add_dev_path():
    """
    'dev' 폴더 경로를 sys.path 에 추가합니다.
    """
    current_path = os.path.abspath(__file__)
    while True:
        current_path = os.path.dirname(current_path)
        if os.path.basename(current_path) == "dev":
            if current_path not in sys.path:
                sys.path.insert(0, current_path)
            return current_path
        if current_path == os.path.dirname(current_path):
            raise RuntimeError("'dev' 폴더를 찾을 수 없습니다.")

def add_cuda_dll_directory():
    """
    CUDA DLL이 있는 경로를 환경 변수에 추가합니다.
    """
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
    if os.path.exists(cuda_path):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(cuda_path)
        else:
            os.environ["PATH"] = cuda_path + os.pathsep + os.environ["PATH"]
        print("✅ CUDA DLL 경로 등록됨:", cuda_path)
    else:
        print("⚠️ CUDA 경로를 찾을 수 없습니다:", cuda_path)

def add_all_pyd_build_paths():
    """
    dev/backend 내부의 모든 build/lib.* 디렉토리를 찾아 sys.path에 추가합니다.
    """
    dev_path = find_and_add_dev_path()
    backend_path = os.path.join(dev_path, "backend")
    if not os.path.exists(backend_path):
        print("❌ backend 디렉토리가 존재하지 않음:", backend_path)
        return

    for root, dirs, files in os.walk(backend_path):
        for dir_name in dirs:
            if dir_name == "build":
                build_dir = os.path.join(root, dir_name)
                for sub_dir in os.listdir(build_dir):
                    if sub_dir.startswith("lib.win"):
                        full_path = os.path.join(build_dir, sub_dir)
                        if os.path.exists(full_path) and full_path not in sys.path:
                            sys.path.append(full_path)
                            print("🔧 PYD 경로 추가됨:", full_path)

def setup_paths():
    ensure_project_root_in_sys_path()
    find_and_add_dev_path()
    add_cuda_dll_directory()
    add_all_pyd_build_paths()

def import_cuda_module(module_name, build_dir):
    """
    지정된 CUDA 모듈을 import합니다.
    
    Args:
        module_name (str): import할 모듈 이름 (예: 'losses_cuda')
        build_dir (str): .pyd가 들어있는 디렉토리 경로
    Returns:
        module: import된 모듈 객체
    """
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")

    if build_dir not in sys.path:
        sys.path.insert(0, build_dir)
        print(f"✅ .pyd 경로 등록됨: {build_dir}")

    try:
        mod = __import__(module_name)
        print(f"✅ CUDA 모듈 '{module_name}' import 성공!")
        return mod
    except ImportError as e:
        print(f"❌ CUDA 모듈 '{module_name}' import 실패:", e)
        raise


