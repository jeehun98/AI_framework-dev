@echo off
set BUILD_TYPE=%1
if "%BUILD_TYPE%"=="" set BUILD_TYPE=Release
if not exist build mkdir build
cd build
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ..
cmake --build . --config %BUILD_TYPE%
