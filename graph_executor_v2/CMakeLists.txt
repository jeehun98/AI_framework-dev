cmake_minimum_required(VERSION 3.24)

project(graph_executor_v2 LANGUAGES CXX CUDA)

# =========================
# 기본 설정
# =========================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA 아키텍처 지정 (예: 80;86;89;90)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)  # 기본값: Ampere(8.6)
endif()

option(BUILD_PYTHON "Build Python extension module (_core)" ON)

# =========================
# Python / pybind11 / CUDA Toolkit
# =========================
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)   # CUDA::cudart, CUDA::cublas 등 제공

# =========================
# include 경로
# =========================
set(PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(REGEMM_INCLUDE_DIR  ${CMAKE_SOURCE_DIR}/kernels/regemm_epilogue/include)

include_directories(${PROJECT_INCLUDE_DIR})
include_directories(${REGEMM_INCLUDE_DIR})

# =========================
# ai_core (디스패치/실행기/ops)
# =========================
add_library(ai_core STATIC
  src/dispatch/registry.cpp
  src/dispatch/selector_rules.cpp
  src/executor/scheduler.cpp
  src/executor/mem_planner.cpp
  src/executor/autograd_engine.cpp
  src/ops/gemm.cpp
)

target_include_directories(ai_core PUBLIC
  ${PROJECT_INCLUDE_DIR}
)

# =========================
# ai_backend_cuda (CUDA 백엔드 + regemm 커널들)
# =========================
add_library(ai_backend_cuda STATIC
  backends/cuda/register_ops.cpp
  backends/cuda/ops/gemm/launcher.cu
  backends/cuda/ops/gemm/backward.cu

  # regemm 커널/런처
  # kernels/regemm_epilogue/src/launcher.cu
  kernels/regemm_epilogue/src/regemm_gemm_bias_act.cu
  kernels/regemm_epilogue/src/regemm_backward.cu
)

# 필요시 분리 컴파일(링크 타임 최적화 이슈 회피)
set_target_properties(ai_backend_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(ai_backend_cuda PUBLIC
  ${PROJECT_INCLUDE_DIR}
  ${REGEMM_INCLUDE_DIR}
)

# cuBLAS & cudart 링크
target_link_libraries(ai_backend_cuda
  PUBLIC
    ai_core
    CUDA::cudart
    CUDA::cublas
)

# =========================
# Python 모듈 (_core) — pybind11
# =========================
if(BUILD_PYTHON)
  pybind11_add_module(_core MODULE
    src/bindings/py_api.cpp
  )
  target_link_libraries(_core PRIVATE ai_core ai_backend_cuda)

  # 빌드 산출물 위치를 패키지 폴더로 지정하여 바로 import 가능
  set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2"
  )
  if (WIN32)
    # Windows에서도 .pyd가 python/graph_executor_v2/에 위치
    set_target_properties(_core PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2"
    )
  endif()
endif()

# =========================
# 경고 레벨(컴파일 옵션)
# =========================
if (MSVC)
  # ai_core는 순수 C++이므로 일반 경고 옵션 적용
  target_compile_options(ai_core PRIVATE /W4)
else()
  target_compile_options(ai_core PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
  target_compile_options(ai_backend_cuda PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
endif()

# =========================
# NVCC <=MSVC 콤마 파싱 이슈 회피
#  - CUDA 컴파일 시 PDB 옵션(/Fd 등) 비활성화
#  - -Xcompiler 옵션을 각각 분리하여 전달
# =========================
if (MSVC AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  # CUDA 컴파일 단계 PDB 생성 비활성화 (Release 계열에서 불필요)
  set(CMAKE_CUDA_COMPILE_PDB_NAME "")
  set(CMAKE_CUDA_COMPILE_PDB_OUTPUT_DIRECTORY "")

  # CUDA 컴파일에 붙을 MSVC 옵션은 '각각'의 -Xcompiler로 전달 (콤마로 묶지 않음)
  target_compile_options(ai_backend_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/MD>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/O2>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/Ob2>
    # 정말 필요할 때만: $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/FS>
  )

  # (선택) Debug/RelWithDebInfo 에서만 PDB 유지
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
      "$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>")
endif()

# =========================
# 상태 출력
# =========================
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
