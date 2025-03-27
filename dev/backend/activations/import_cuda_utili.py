import os
import sys
import importlib.util

def import_cuda_module(module_name: str, search_root: str):
    """
    CUDA로 빌드된 .pyd 모듈을 지정된 경로에서 찾아 동적으로 import합니다.

    Args:
        module_name (str): 모듈 이름 (예: 'activations_cuda')
        search_root (str): .pyd 파일을 검색할 루트 폴더 경로

    Returns:
        모듈 객체 (import된 결과)
    """
    # .pyd 확장 모듈명 찾기
    for root, _, files in os.walk(search_root):
        for file in files:
            if file.startswith(module_name) and file.endswith(".pyd"):
                pyd_path = os.path.join(root, file)
                print(f"✅ 모듈 찾음: {pyd_psath}")

                # 경로를 sys.path에 추가
                sys.path.append(root)

                # 동적 import
                spec = importlib.util.spec_from_file_location(module_name, pyd_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

    raise ImportError(f"❌ {module_name} 모듈(.pyd)을 {search_root} 아래에서 찾을 수 없습니다.")
