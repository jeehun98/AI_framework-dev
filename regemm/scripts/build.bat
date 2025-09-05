@echo off
setlocal
mkdir build
cd build
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release
ninja
ctest --output-on-failure
endlocal
