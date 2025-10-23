
# CMake configure 단계 - 데탑
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe" -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6" -DCMAKE_CUDA_ARCHITECTURES=86 -DGE2_WITH_REGEMM=ON -DPython_EXECUTABLE="C:/Users/as042/AppData/Local/Programs/Python/Python312/python.exe" -DPython_ROOT_DIR="C:/Users/as042/AppData/Local/Programs/Python/Python312" -Dpybind11_DIR="C:/Users/as042/AppData/Local/Programs/Python/Python312/Lib/site-packages/pybind11/share/cmake/pybind11"

 
# 노트북
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DGE2_WITH_REGEMM=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DPython3_EXECUTABLE="C:\Users\owner\AppData\Local\Programs\Python\Python312\python.exe" -Dpybind11_DIR="C:\Users\owner\AppData\Local\Programs\Python\Python312\Lib\site-packages\pybind11\share\cmake\pybind11"

# nvtx 추가 버전
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DUSE_NVTX=ON -DGE2_WITH_REGEMM=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DPython3_EXECUTABLE="C:\Users\owner\AppData\Local\Programs\Python\Python312\python.exe" -Dpybind11_DIR="C:\Users\owner\AppData\Local\Programs\Python\Python312\Lib\site-packages\pybind11\share\cmake\pybind11"


# 새로 추가된 op 전용 바인딩만:
cmake --build build --target _ops_gemm -j

# 기존 코어 확장 모듈:
cmake --build build --target _core -j
