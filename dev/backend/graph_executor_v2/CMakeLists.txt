# CMakeLists.txt
#
# 목적
#  - pybind11 기반의 파이썬 확장 모듈(graph_executor_v2) 빌드
#  - CUDA / cuBLAS / cuBLASLt 연동
#  - Windows / Linux 공용 설정
#
# 빌드 결과
#  - Python 모듈 파일: graph_executor_v2.(pyd|so)
#  - include/ge_v2_api.h 설치(optional)

cmake_minimum_required(VERSION 3.26)

# 프로젝트 이름/언어
project(graph_executor_v2 LANGUAGES CXX CUDA)

# ------------------------------ Python / pybind11 ------------------------------
# Python 해더/링커(Development.Module) 필요
find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)

# pybind11을 우선 시스템에서 찾고, 없으면 FetchContent로 가져온다.
find_package(pybind11 CONFIG QUIET)
if (NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.12.0  # 모듈 네이밍/ABI 안정적인 버전
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# ------------------------------ CUDA / cuBLAS(Lt) ------------------------------
find_package(CUDAToolkit REQUIRED)

# 목적 GPU 아키텍처 지정
#   - 예) Ampere(86) / Ada(89) 등. 여럿 지정 가능("86;89")
#   - 환경 따라 -DCMAKE_CUDA_ARCHITECTURES="86;89" 로 오버라이드 권장.
set(CMAKE_CUDA_ARCHITECTURES 86)

# ------------------------------ 소스 목록 ------------------------------
set(SRC
  src/bindings_min_api.cpp   # pybind11 바인딩(launch_kernel 등 노출)
  src/launch_table.cpp       # 커널/캡빌리티 테이블
  src/my_kernels.cu          # CUDA 커널/ cuBLASLt 호출부
)

# ------------------------------ 파이썬 모듈 타겟 ------------------------------
# pybind11_add_module:
#  - 파이썬 확장 모듈 빌드에 필요한 컴파일/링크 규칙을 자동 설정.
#  - MODULE 타입의 공유 라이브러리를 만든다.
pybind11_add_module(graph_executor_v2 MODULE ${SRC})

# C++/CUDA 표준
target_compile_features(graph_executor_v2 PRIVATE cxx_std_17)
set_target_properties(graph_executor_v2 PROPERTIES
  CUDA_STANDARD 17
  CUDA_SEPARABLE_COMPILATION ON     # NVCC 분리 컴파일(링크 타임 최적화에 유리)
  POSITION_INDEPENDENT_CODE ON      # .so/.pyd에 필요
  OUTPUT_NAME "graph_executor_v2"   # 파이썬 모듈 이름
)

# ------------------------------ 컴파일 옵션 ------------------------------
# 주의: NVCC는 호스트 컴파일러 옵션을 -Xcompiler 로 전달해야 한다.
if (MSVC)
  target_compile_options(graph_executor_v2 PRIVATE
    # C++ 컴파일러(MSVC) 경고/최적화/런타임 라이브러리
    $<$<COMPILE_LANGUAGE:CXX>:/bigobj;/W3;/EHsc;/O2;/MD>
  )
else()
  target_compile_options(graph_executor_v2 PRIVATE
    # GCC/Clang 공통 옵션
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-O2>
  )
endif()

# CUDA 소스에 대해 호스트 컴파일러 옵션 전달
target_compile_options(graph_executor_v2 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc;-Xcompiler=/O2;-Xcompiler=/MD>
)
# 참고:
#  - /MD: 멀티스레드 DLL CRT 사용(윈도우). 리눅스에선 무시됨.
#  - /EHsc: C++ 예외 모델. NVCC가 MSVC를 호출할 때 필요.

# ------------------------------ 링크 라이브러리 ------------------------------
# * cudart: CUDA 런타임
# * cublas, cublasLt: GEMM 및 에필로그 지원
target_link_libraries(graph_executor_v2 PRIVATE
  CUDA::cudart
  CUDA::cublas
  CUDA::cublasLt
)

# ------------------------------ include 경로 ------------------------------
target_include_directories(graph_executor_v2 PRIVATE
  ${CUDAToolkit_INCLUDE_DIRS}  # cuda_runtime_api.h 등
  include                      # ge_v2_api.h
)

# ------------------------------ 설치 규칙(선택) ------------------------------
# 파이썬 패키징(PEP 517/518)이나 수동 배포 시 활용 가능.
install(TARGETS graph_executor_v2
        LIBRARY DESTINATION .   # *.so (Linux/macOS)
        RUNTIME DESTINATION .   # *.pyd (Windows)
        ARCHIVE DESTINATION .)  # 정적 아카이브가 생성될 경우

install(FILES include/ge_v2_api.h DESTINATION include)

# ------------------------------ 빌드 참고 ------------------------------
# 1) 빌드 예시(Windows PowerShell):
#    cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
#          -DPython_ROOT_DIR="C:\Python312" `
#          -DCMAKE_CUDA_ARCHITECTURES=86
#    cmake --build build --config Release
#
# 2) 빌드 예시(Linux):
#    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="86;89"
#    cmake --build build -j
#
# 3) 런타임 주의:
#    - Windows: cudart64_*.dll, cublas*.dll, cublasLt*.dll 이 PATH에 있어야 함.
#    - Linux: 보통 /usr/lib/x86_64-linux-gnu 등에 설치. 미설치면 LD_LIBRARY_PATH 필요.
