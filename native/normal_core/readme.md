# 📦 normal_core Build & Usage Guide

## 1. 프로젝트 개요

`normal_core`는 **정규분포 난수 생성 모듈**입니다.\
- CPU 구현: `std::mt19937_64 + std::normal_distribution`\
- CUDA 구현: `xorshift128+ + Box–Muller` (CURAND 의존성 없음)\
- API는 `include/normal/normal.hpp`에 정의되며, CPU/CUDA 양쪽에서 동일
인터페이스를 제공합니다.

## 2. 디렉토리 구조

    normal_core
     ┣ 📂examples
     ┃ ┗ 📜main.cpp           # 예제 실행 파일
     ┣ 📂include
     ┃ ┗ 📂normal
     ┃    ┣ 📜normal.hpp      # 공개 API
     ┃    ┗ 📜normal_export.h # DLL export 매크로
     ┣ 📂src
     ┃ ┣ 📜normal_cpu.cpp     # CPU 구현
     ┃ ┗ 📜normal_cuda.cu     # CUDA 구현
     ┗ 📜CMakeLists.txt       # 빌드 스크립트

## 3. 필수 환경

-   **Windows 10/11 (x64)**
-   **Visual Studio 2022** (Desktop development with C++)
-   **CMake ≥ 3.22**
-   **Ninja**
-   **CUDA Toolkit 12.x** (옵션: CUDA 백엔드 사용 시)

> Visual Studio 설치 시 반드시 **Windows 10/11 SDK**와 `MSVC v143`
> 이상을 포함해야 합니다.

## 4. CMake 주요 옵션

  -------------------------------------------------------------------------------
  옵션                         기본값                       설명
  ---------------------------- ---------------------------- ---------------------
  `BUILD_SHARED_LIBS`          `ON`                         DLL/so로 빌드 여부

  `NORMAL_WITH_CUDA`           `OFF`                        CUDA 백엔드 빌드 여부

  `CMAKE_CUDA_ARCHITECTURES`   없음(자동)                   GPU 아키텍처 (예:
                                                            86=RTX30 시리즈)
  -------------------------------------------------------------------------------

## 5. 빌드 방법

### CPU 전용 빌드

``` powershell
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

빌드 결과: - `build/normal_core.dll` (+ `.lib`) -
`build/normal_example.exe`

### CUDA 포함 빌드

``` powershell
cmake -S . -B build_cuda -G "Ninja" -DCMAKE_BUILD_TYPE=Release ^
  -DNORMAL_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build_cuda --config Release -v
```

빌드 결과: - `build_cuda/normal_core.dll` -
`build_cuda/normal_example.exe`

## 6. 실행 예제

``` powershell
cd build_cuda
.
ormal_example.exe
```

예상 출력:

    [CPU] N=1048576 time=12.3 ms  mean=2.0007 std=2.998
    [CUDA] N=1048576 time=3.1 ms  mean=1.9998 std=3.002

-   CPU와 CUDA 모두 평균≈2.0, 표준편차≈3.0 근처 값이 나오면 정상입니다.\
-   CUDA 실행 시 `cudart64_*.dll`이 PATH에 있어야 합니다.

## 7. 트러블슈팅

### (1) `CMake Error: CMAKE_MT-NOTFOUND`

-   Windows SDK가 설치되지 않았거나, 개발자 프롬프트를 쓰지 않은 경우
    발생\
-   해결: Visual Studio Installer → `Windows 10/11 SDK` 설치\
-   `x64 Native Tools Command Prompt for VS 2022` 실행 후 다시 빌드

### (2) `C2146`, `C2882` (std 앞 구문 오류)

-   원인: 소스 인코딩 문제(UTF-8 미설정)\
-   해결: `CMakeLists.txt`에 `/utf-8` 옵션 추가 완료 (이미 반영됨)

### (3) `fatal error: 'normal/normal.hpp': No such file or directory`

-   원인: nvcc의 include 경로 누락\
-   해결: `target_include_directories(normal_cuda PRIVATE include)` 추가
    (이미 반영됨)

### (4) `LNK2019 unresolved external symbol __imp_generate_cuda`

-   원인: `NORMAL_WITH_CUDA`/`NORMAL_EXPORTS` 매크로가 CUDA 빌드 타깃에
    정의되지 않음\
-   해결:
    `target_compile_definitions(normal_cuda PRIVATE NORMAL_WITH_CUDA NORMAL_EXPORTS)`
    추가 (이미 반영됨)

## 8. 다음 단계

-   **Pybind11 바인딩**: Python에서 CPU/GPU 난수 호출 지원
-   **성능 최적화**: CPU OpenMP 병렬화, CUDA 블록 크기 튜닝
-   **테스트 모듈 추가**: GoogleTest 기반 단위 테스트
