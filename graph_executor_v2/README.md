
# CMake configure 단계 - 데탑
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe" -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6" -DCMAKE_CUDA_ARCHITECTURES=86 -DGE2_WITH_REGEMM=ON -DPython_EXECUTABLE="C:/Users/as042/AppData/Local/Programs/Python/Python312/python.exe" -DPython_ROOT_DIR="C:/Users/as042/AppData/Local/Programs/Python/Python312" -Dpybind11_DIR="C:/Users/as042/AppData/Local/Programs/Python/Python312/Lib/site-packages/pybind11/share/cmake/pybind11"

cmake --build build --config Release -j


# 기존 build 폴더 삭제
rmdir /s /q build

# CMake configure 단계
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DGE2_WITH_REGEMM=ON -DCMAKE_CUDA_ARCHITECTURES=86  -DPython_EXECUTABLE="C:\Users\owner\AppData\Local\Programs\Python\Python312\python.exe" -Dpybind11_DIR="C:\Users\owner\AppData\Local\Programs\Python\Python312\Lib\site-packages\pybind11\share\cmake\pybind11"


# 실제 빌드 단계
cmake --build build -j

 