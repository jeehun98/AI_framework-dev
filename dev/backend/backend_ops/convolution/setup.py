from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='convolution_cuda',
    ext_modules=[
        CUDAExtension(
            name='convolution_cuda',
            sources=['convolution_cuda.cu'],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': [
                    '-O2',
                    '--extended-lambda',
                    '-gencode=arch=compute_70,code=sm_70',  # 적절한 compute capability로 변경
                    '-gencode=arch=compute_75,code=sm_75',
                    '-Xcompiler', '/MD',
                    '-std=c++17'
                ],
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
