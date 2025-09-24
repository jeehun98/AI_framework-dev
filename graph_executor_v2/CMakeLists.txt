cmake_minimum_required(VERSION 3.24)

project(graph_executor_v2 LANGUAGES CXX CUDA)

# =========================
# 기본 설정
# =========================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# IDE/IntelliSense 정확도 향상을 위해 컴파일 DB 출력
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
# include 경로 변수
# =========================
set(PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(REGEMM_INCLUDE_DIR  ${CMAKE_SOURCE_DIR}/kernels/regemm_epilogue/include)

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
  src/ops/rmsnorm.cpp
  src/ops/softmax.cpp
  src/ops/layernorm.cpp
  src/ops/cross_entropy.cpp
  src/ops/dropout.cpp
  src/ops/sdpa.cpp
  src/ops/conv2d.cpp
  src/ops/pool2d.cpp
  src/ops/elementwise.cpp
  src/ops/reduction.cpp
  src/ops/concat.cpp
)

target_include_directories(ai_core
  PUBLIC
    ${PROJECT_INCLUDE_DIR}     # graph_executor_v2/include
  PRIVATE
    ${CMAKE_SOURCE_DIR}        # graph_executor_v2/ (backends/... 등의 루트 기준 include 허용)
)

if (MSVC)
  target_compile_options(ai_core PRIVATE /W4)
else()
  target_compile_options(ai_core PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
endif()

# =========================
# ai_backend_cuda (CUDA 백엔드 + 내부 regemm 커널들)
# =========================
add_library(ai_backend_cuda STATIC
  backends/cuda/register_ops.cpp
  backends/cuda/ops/gemm/launcher.cu
  backends/cuda/ops/gemm/backward.cu

  backends/cuda/ops/rmsnorm/kernels.cu
  backends/cuda/ops/rmsnorm/launcher.cu

  backends/cuda/ops/layernorm/kernels.cu
  backends/cuda/ops/layernorm/launcher.cu

  backends/cuda/ops/softmax/kernels.cu
  backends/cuda/ops/softmax/launcher.cu

  backends/cuda/ops/cross_entropy/kernels.cu
  backends/cuda/ops/cross_entropy/launcher.cu

  backends/cuda/ops/dropout/kernels.cu
  backends/cuda/ops/dropout/launcher.cu

  backends/cuda/ops/conv2d/kernels.cu
  backends/cuda/ops/conv2d/launcher.cu

  backends/cuda/ops/pool2d/kernels.cu
  backends/cuda/ops/pool2d/launcher.cu

  backends/cuda/ops/sdpa/launcher.cu

  backends/cuda/ops/elementwise/kernels.cu
  backends/cuda/ops/elementwise/launcher.cu

  backends/cuda/ops/reduction/kernels.cu
  backends/cuda/ops/reduction/launcher.cu

  backends/cuda/ops/concat/launcher.cu

  # 내부(regemm_epilogue) 커널
  kernels/regemm_epilogue/src/regemm_gemm_bias_act.cu
  kernels/regemm_epilogue/src/regemm_backward.cu
)

# 분리 컴파일(링크 타임 device code 결합 안정화)
set_target_properties(ai_backend_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(ai_backend_cuda
  PUBLIC
    ${PROJECT_INCLUDE_DIR}
    ${REGEMM_INCLUDE_DIR}
  PRIVATE
    ${CMAKE_SOURCE_DIR}        # 루트 기준 include 허용
)

# cuBLAS & cudart 링크
target_link_libraries(ai_backend_cuda
  PUBLIC
    ai_core
    CUDA::cudart
    CUDA::cublas
)

# CUDA 컴파일러 옵션
target_compile_options(ai_backend_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=[compute_${CMAKE_CUDA_ARCHITECTURES},sm_${CMAKE_CUDA_ARCHITECTURES}]>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc>
)

if (MSVC AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  # NVCC <=MSVC 콤마 파싱/PDB 옵션 이슈 회피
  set(CMAKE_CUDA_COMPILE_PDB_NAME "")
  set(CMAKE_CUDA_COMPILE_PDB_OUTPUT_DIRECTORY "")

  # CUDA 단계에 붙는 MSVC 옵션은 각각의 -Xcompiler로 전달
  target_compile_options(ai_backend_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/MD>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/O2>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/Ob2>
    # 필요 시만: $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/FS>
  )

  # (선택) Debug/RelWithDebInfo 에서만 PDB 유지
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
      "$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>")
else()
  target_compile_options(ai_backend_cuda PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
endif()

# =========================
# Python 모듈 (_core) — pybind11
# =========================
if(BUILD_PYTHON)
  pybind11_add_module(_core MODULE
    src/bindings/py_api.cpp
  )
  target_link_libraries(_core PRIVATE ai_core ai_backend_cuda)

  target_include_directories(_core
    PRIVATE
      ${PROJECT_INCLUDE_DIR}
      ${CMAKE_SOURCE_DIR}      # 루트 기준 include 허용
  )

  # 빌드 산출물 위치를 패키지 폴더로 지정하여 바로 import 가능
  set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2"
  )
  if (WIN32)
    set_target_properties(_core PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2"
    )
  endif()
endif()

# =========================
# 상태 출력
# =========================
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "PROJECT_INCLUDE_DIR: ${PROJECT_INCLUDE_DIR}")
message(STATUS "REGEMM_INCLUDE_DIR: ${REGEMM_INCLUDE_DIR}")
