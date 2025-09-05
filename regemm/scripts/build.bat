@echo off
setlocal
mkdir build
cd build
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release  -DCMAKE_CUDA_ARCHITECTURES=86
ninja
ctest --output-on-failure
endlocal
