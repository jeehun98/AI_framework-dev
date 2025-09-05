:: x64 개발자 환경 열기 (한 번만)
"%VSINSTALLDIR%Common7\Tools\VsDevCmd.bat" -arch=x64

:: 깨끗한 빌드
cd C:\Users\owner\Desktop\AI_framework-dev\regemm
rmdir /s /q build & mkdir build & cd build
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release
ninja
ctest --output-on-failure

:: 벤치마크
.\bench_regemm 2048 2048 2048 30
