import sys
import os
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# --------- 유틸 ---------
def abspaths(paths):
    return [os.path.abspath(p) for p in paths]

# --------- 커스텀 Extension ---------
class CUDAExtension(Extension):
    def __init__(self, name, sources):
        super().__init__(name, sources)

# --------- 커스텀 build_ext ---------
class BuildExtWithNvcc(build_ext):
    def build_extension(self, ext):
        if isinstance(ext, CUDAExtension):
            self.build_cuda_extension(ext)
        else:
            super().build_extension(ext)

    def build_cuda_extension(self, ext: CUDAExtension):
        sources = abspaths(ext.sources)
        output_file = os.path.abspath(self.get_ext_fullpath(ext.name))
        build_dir = os.path.dirname(output_file)
        os.makedirs(build_dir, exist_ok=True)

        # ----- 경로 설정 -----
        # pybind11 include
        try:
            import pybind11
            pybind_include = pybind11.get_include()
        except Exception:
            # 폴백: 사용자가 알려준 경로
            pybind_include = r"C:\Users\as042\AppData\Local\Programs\Python\Python312\Lib\site-packages\pybind11\include"

        # Python include / libs
        python_include = r"C:\Users\as042\AppData\Local\Programs\Python\Python312\include"
        python_lib     = r"C:\Users\as042\AppData\Local\Programs\Python\Python312\libs"

        # CUDA 경로 (환경에 맞게 수정 가능; CUDA_PATH가 있으면 우선 사용)
        cuda_home = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6")
        cuda_include_path = os.path.join(cuda_home, "include")
        cuda_lib_path     = os.path.join(cuda_home, "lib", "x64")
        cuda_bin_path     = os.path.join(cuda_home, "bin")

        # 실행 시 DLL 찾도록 PATH 보강
        os.environ["PATH"] = cuda_bin_path + os.pathsep + os.environ.get("PATH", "")

        # ----- NVCC 명령 구성 -----
        # 주의: shell=False로 리스트 전달 → 공백 포함 경로도 안전
        nvcc_cmd = [
            "nvcc",
            "-O2",
            "-Xcompiler", "/MD",
            "-shared",
            "-x", "cu",
            # (옵션) 아키텍처 지정하고 싶다면 아래 주석 해제
            # "-gencode=arch=compute_86,code=sm_86",
            f"-I{pybind_include}",
            f"-I{python_include}",
            f"-I{cuda_include_path}",
            "-o", output_file,
        ] + sources + [
            f"-L{python_lib}",
            f"-L{cuda_lib_path}",
            "-lcudart",
            "-lcublas",       # ✅ cuBLAS 링크 (필수)
            # "-lcublasLt",   # (선택) cuBLASLt 사용 시 주석 해제
            "-lpython312",
        ]

        try:
            subprocess.check_call(nvcc_cmd, shell=False)
        except subprocess.CalledProcessError as e:
            print("\n❌ NVCC command failed with exit code:", e.returncode)
            print("Command was:\n", " ".join(nvcc_cmd))
            raise

# --------- 커맨드라인에서 --name / --sources 파싱 ---------
def get_extension():
    if "--name" not in sys.argv or "--sources" not in sys.argv:
        print("Usage: python setup.py build_ext --inplace --name <module_name> --sources <file1> <file2> ...")
        sys.exit(1)

    name_index = sys.argv.index("--name") + 1
    sources_index = sys.argv.index("--sources") + 1

    module_name = sys.argv[name_index]
    source_files = sys.argv[sources_index:]

    # setuptools 인자 목록에서 커스텀 인자 제거
    sys.argv = sys.argv[:name_index - 1] + sys.argv[name_index + 1:sources_index - 1]

    return CUDAExtension(module_name, source_files)

# --------- setup() ---------
setup(
    name="graph_executor",
    version="0.1",
    ext_modules=[get_extension()],
    cmdclass={"build_ext": BuildExtWithNvcc},
    zip_safe=False,
)
