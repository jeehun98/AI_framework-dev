from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# 명시적으로 환경 변수에 CUDA 및 MSVC 경로 추가
os.environ['PATH'] += ';C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin'
os.environ['PATH'] += ';C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.41.34120/bin/Hostx64/x64'

setup(
    name='kernel',
    ext_modules=[
        CUDAExtension(
            name='kernel',
            sources=['kernel.cu'],
            extra_compile_args={
                'nvcc': [
                    '-O2',  # 최적화 플래그
                    '--verbose',  # 상세 로그 출력
                    '-gencode=arch=compute_86,code=sm_86',  # GPU 아키텍처 설정
                    '--compiler-bindir', 
                    'C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.41.34120/bin/Hostx64/x64'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
