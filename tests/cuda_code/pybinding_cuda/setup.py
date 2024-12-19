from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os

class BuildExtWithNvcc(build_ext):
    def build_extensions(self):
        for ext in self.extensions:
            if isinstance(ext, CUDAExtension):
                self.build_cuda_extension(ext)
            else:
                super().build_extensions()

    def build_cuda_extension(self, ext):
        sources = [os.path.abspath(src) for src in ext.sources]
        output_file = os.path.abspath(self.get_ext_fullpath(ext.name))
        build_dir = os.path.dirname(output_file)
        
        # Pybind11 및 Python 경로 설정
        pybind_include = r"C:\Users\owner\AppData\Local\Programs\Python\Python312\Lib\site-packages\pybind11\include"
        python_include = r"C:\Users\owner\AppData\Local\Programs\Python\Python312\include"
        python_lib = r"C:\Users\owner\AppData\Local\Programs\Python\Python312\libs"

        # CUDA 경로 설정
        cuda_lib_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64"
        cuda_bin_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"

        # 출력 디렉토리 생성
        os.makedirs(build_dir, exist_ok=True)

        # NVCC 명령 설정
        nvcc_cmd = [
            "nvcc",
            "-shared",                      # 공유 라이브러리 생성
            "-O2",                          # 최적화
            "-x", "cu",                     # CUDA 코드
            f'-I"{pybind_include}"',        # Pybind11 헤더 경로
            f'-I"{python_include}"',        # Python 헤더 경로
            f'-L"{python_lib}"',            # Python 라이브러리 경로 추가
            f'-L"{cuda_lib_path}"',         # CUDA 라이브러리 경로 추가
            "--compiler-options", "/MD",    # MSVC 컴파일러 옵션
            "-lcudart",                     # CUDA 런타임 라이브러리
            "-lpython312",                  # Python 라이브러리 연결
            "-o", f'"{output_file}"',       # 출력 파일 설정
        ] + sources

        # 링커 로그 활성화 (디버깅용)
        nvcc_cmd.extend(["--verbose", "-Xlinker", "/VERBOSE"])

        print("Running NVCC:", " ".join(nvcc_cmd))

        # CUDA DLL 경로를 PATH에 추가
        os.environ["PATH"] += os.pathsep + cuda_bin_path

        # NVCC 실행
        subprocess.check_call(" ".join(nvcc_cmd), shell=True)

class CUDAExtension(Extension):
    def __init__(self, name, sources):
        super().__init__(name, sources)
        self.sources = sources

setup(
    name="cuda_add_example",
    version="0.1",
    ext_modules=[
        CUDAExtension("cuda_add", ["cuda_add.cu"])  # CUDA 확장 모듈
    ],
    cmdclass={"build_ext": BuildExtWithNvcc},       # 빌드 확장 클래스 등록
    zip_safe=False,
)
