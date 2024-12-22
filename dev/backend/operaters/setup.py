import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import subprocess

# 빌드 확장 클래스
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
        pybind_include = r"C:\\Users\\owner\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pybind11\\include"
        python_include = r"C:\\Users\\owner\\AppData\\Local\\Programs\\Python\\Python312\\include"
        python_lib = r"C:\\Users\\owner\\AppData\\Local\\Programs\\Python\\Python312\\libs"

        # CUDA 경로 설정
        cuda_lib_path = r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\lib\\x64"
        cuda_bin_path = r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\bin"

        # 출력 디렉토리 생성
        os.makedirs(build_dir, exist_ok=True)

        # NVCC 명령 설정
        nvcc_cmd = [
            "nvcc",
            "-shared",
            "-O2",
            "-x", "cu",
            f'-I"{pybind_include}"',
            f'-I"{python_include}"',
            f'-L"{python_lib}"',
            f'-L"{cuda_lib_path}"',
            "--compiler-options", "/MD",
            "-lcudart",
            "-lpython312",
            "-o", f'"{output_file}"',
        ] + sources

        print("Running NVCC:", " ".join(nvcc_cmd))

        os.environ["PATH"] += os.pathsep + cuda_bin_path

        try:
            subprocess.check_call(" ".join(nvcc_cmd), shell=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during NVCC compilation: {e}")
            sys.exit(1)

# CUDA 확장 클래스
class CUDAExtension(Extension):
    def __init__(self, name, sources):
        super().__init__(name, sources)
        self.sources = sources

# 확장 모듈 동적 설정
def get_extension():
    if "--name" not in sys.argv or "--sources" not in sys.argv:
        print("Usage: python setup.py build_ext --name <module_name> --sources <source_file1> <source_file2> ...")
        sys.exit(1)

    # 명령줄 인자 처리
    name_index = sys.argv.index("--name") + 1
    sources_index = sys.argv.index("--sources") + 1

    if name_index >= len(sys.argv) or sources_index >= len(sys.argv):
        print("Error: --name and --sources arguments are required.")
        sys.exit(1)

    module_name = sys.argv[name_index]
    source_files = sys.argv[sources_index:]

    # 명령줄에서 --name 및 --sources 제거
    sys.argv = sys.argv[:name_index - 1] + sys.argv[name_index + 1:sources_index - 1]

    return CUDAExtension(module_name, source_files)

setup(
    name="dynamic_cuda_builder",
    version="0.1",
    ext_modules=[get_extension()],
    cmdclass={"build_ext": BuildExtWithNvcc},
    zip_safe=False,
)
