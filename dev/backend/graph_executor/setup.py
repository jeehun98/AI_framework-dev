import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import subprocess

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

        pybind_include = r"C:\Users\as042\AppData\Local\Programs\Python\Python312\Lib\site-packages\pybind11\include"
        python_include = r"C:\Users\as042\AppData\Local\Programs\Python\Python312\include"
        python_lib = r"C:\Users\as042\AppData\Local\Programs\Python\Python312\libs"
        cuda_lib_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64"
        cuda_bin_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"


        os.makedirs(build_dir, exist_ok=True)

        nvcc_cmd = [
            "nvcc", "-O2", "-Xcompiler", "/MD", "-shared", "-x", "cu",
            f'-I"{pybind_include}"', f'-I"{python_include}"',
            f'-L"{python_lib}"', f'-L"{cuda_lib_path}"',
            "-lcudart", "-lpython312",
            "-o", f'"{output_file}"'
        ] + sources

        os.environ["PATH"] += os.pathsep + cuda_bin_path
        try:
            subprocess.check_call(" ".join(nvcc_cmd), shell=True)
        except subprocess.CalledProcessError as e:
            print("\n❌ NVCC command failed with exit code:", e.returncode)
            print("Command was:\n", " ".join(nvcc_cmd))
            print("\n--- NVCC output begins ---")
            subprocess.run(" ".join(nvcc_cmd), shell=True)
            print("--- NVCC output ends ---\n")
            raise


class CUDAExtension(Extension):
    def __init__(self, name, sources): super().__init__(name, sources)

def get_extension():
    if "--name" not in sys.argv or "--sources" not in sys.argv:
        print("Usage: python setup.py build_ext --name <module_name> --sources <file1> <file2> ...")
        sys.exit(1)
    name_index = sys.argv.index("--name") + 1
    sources_index = sys.argv.index("--sources") + 1
    module_name = sys.argv[name_index]
    source_files = sys.argv[sources_index:]
    sys.argv = sys.argv[:name_index - 1] + sys.argv[name_index + 1:sources_index - 1]
    return CUDAExtension(module_name, source_files)

setup(
    name="graph_executor",
    version="0.1",
    ext_modules=[get_extension()],
    cmdclass={"build_ext": BuildExtWithNvcc},
    zip_safe=False,
)
