cmake_minimum_required(VERSION 3.24)

project(graph_executor_v2 LANGUAGES CXX CUDA)

# =========================
# 기본 설정
# =========================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# MSVC 런타임 일관(/MD)
if (MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

# CUDA 아키텍처 지정 (예: 86;89;90)
if (NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

option(BUILD_PYTHON "Build Python extension module (_core)" ON)

# =========================
# Python / pybind11 / CUDA Toolkit
# =========================
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)  # CUDA::cudart, CUDA::cublas 등

# =========================
# include 경로 변수
# =========================
set(PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

# =========================
# 공통 경고/옵션 헬퍼
# =========================
function(apply_win_warnings tgt)
  if (MSVC)
    target_compile_options(${tgt} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/W4 /Zc:__cplusplus /bigobj /permissive- /EHsc>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/permissive->
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc>
    )
    target_compile_definitions(${tgt} PRIVATE NOMINMAX)
  else()
    target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
  endif()
endfunction()

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
  src/ops/slice.cpp
  src/ops/indexing.cpp
  src/ops/pad.cpp
  src/ops/memory.cpp
  src/ops/upsample2d.cpp
  src/ops/view.cpp
)

target_include_directories(ai_core
  PUBLIC
    ${PROJECT_INCLUDE_DIR}
  PRIVATE
    ${CMAKE_SOURCE_DIR}
)
apply_win_warnings(ai_core)

# =========================
# GEMM 공용 오브젝트 라이브러리 (소스 공통)
# =========================
set(GE2_REGEMM_SOURCES
  backends/cuda/ops/gemm/launcher.cu
  backends/cuda/ops/gemm/backward.cu
  backends/cuda/ops/gemm/kernels/regemm_gemm_bias_act.cu
  backends/cuda/ops/gemm/kernels/regemm_backward.cu
)

# (1) 통합 백엔드용: BUILD_STANDALONE_OPS 미정의
add_library(ge2_regemm_core OBJECT ${GE2_REGEMM_SOURCES})
set_target_properties(ge2_regemm_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(ge2_regemm_core
  PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${PROJECT_INCLUDE_DIR}
    backends/cuda/ops/gemm/include     # "regemm/*.h"
)
# 통합 경로는 shim이 아니라 진짜 ai 헤더를 사용하므로 전처리 정의 없음
apply_win_warnings(ge2_regemm_core)

# (2) 독립 모듈용: BUILD_STANDALONE_OPS=1 정의
add_library(ge2_regemm_standalone OBJECT ${GE2_REGEMM_SOURCES})
set_target_properties(ge2_regemm_standalone PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(ge2_regemm_standalone
  PRIVATE
    ${CMAKE_SOURCE_DIR}
    backends/cuda/ops/_common/shim     # ai_shim.hpp
    backends/cuda/ops/gemm/include
)
target_compile_definitions(ge2_regemm_standalone PRIVATE BUILD_STANDALONE_OPS=1)
apply_win_warnings(ge2_regemm_standalone)

# =========================
# ai_backend_cuda (CUDA 백엔드 + 모든 CUDA ops)
#  - 통합 백엔드 경로에서는 ge2_regemm_core 객체 재사용
# =========================
add_library(ai_backend_cuda STATIC
  src/ops/register_ops.cpp           # CUDA 백엔드 op 등록

  # ---- 각 CUDA ops ----
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
  backends/cuda/ops/slice/launcher.cu
  backends/cuda/ops/pad/launcher.cu

  backends/cuda/ops/memory/launcher.cu
  backends/cuda/ops/memory/kernels.cu

  backends/cuda/ops/upsample2d/launcher.cu
  backends/cuda/ops/view/launcher.cu

  backends/cuda/ops/indexing/kernels.cu
  backends/cuda/ops/indexing/launcher.cu

  # GEMM 객체(통합용) 결합 — 중복 소스 나열 금지
  $<TARGET_OBJECTS:ge2_regemm_core>
)

set_target_properties(ai_backend_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_include_directories(ai_backend_cuda
  PUBLIC
    backends/cuda/ops/gemm/include       # regemm/api.h 외부서 필요 시
    ${PROJECT_INCLUDE_DIR}
  PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(ai_backend_cuda
  PUBLIC
    ai_core
    CUDA::cudart
    CUDA::cublas
)
apply_win_warnings(ai_backend_cuda)

# =========================
# Python 모듈 (_core)
# =========================
if (BUILD_PYTHON)
  pybind11_add_module(_core MODULE
    src/bindings/py_api.cpp
  )
  target_link_libraries(_core PRIVATE ai_core ai_backend_cuda)
  target_include_directories(_core PRIVATE ${PROJECT_INCLUDE_DIR} ${CMAKE_SOURCE_DIR})
  set_target_properties(_core PROPERTIES
    PREFIX "" SUFFIX ".pyd"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2"
  )
  if (WIN32)
    set_target_properties(_core PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2"
    )
  endif()
  apply_win_warnings(_core)
endif()

# =========================
# Python 모듈 (_ops_gemm) — 독립 바인딩
#  - ge2_regemm_standalone 객체 재사용
# =========================
pybind11_add_module(_ops_gemm MODULE
  src/bindings/gemm_pybind.cpp
)
target_link_libraries(_ops_gemm PRIVATE
  $<TARGET_OBJECTS:ge2_regemm_standalone>
  CUDA::cudart
  CUDA::cublas
)
target_include_directories(_ops_gemm PRIVATE
  ${CMAKE_SOURCE_DIR}
  backends/cuda/ops/_common/shim      # ai_shim.hpp
  backends/cuda/ops/gemm/include      # regemm headers
)
# 모듈 자체에도 shim 경로 사용 신호(선택적): 필요 없지만 맞춰둠
target_compile_definitions(_ops_gemm PRIVATE BUILD_STANDALONE_OPS=1)

set_target_properties(_ops_gemm PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  PREFIX "" SUFFIX ".pyd"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
)
if (WIN32)
  set_target_properties(_ops_gemm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
  )
endif()
apply_win_warnings(_ops_gemm)

# =========================
# Python 모듈 (_ops_dropout) — 독립 바인딩
# =========================
pybind11_add_module(_ops_dropout MODULE
  src/bindings/dropout_pybind.cpp
  backends/cuda/ops/dropout/launcher.cu
  backends/cuda/ops/dropout/kernels.cu
)
target_compile_definitions(_ops_dropout PRIVATE BUILD_STANDALONE_OPS=1)
target_include_directories(_ops_dropout PRIVATE
  ${CMAKE_SOURCE_DIR}
  backends/cuda/ops/_common/shim
)
target_link_libraries(_ops_dropout PRIVATE CUDA::cudart)
set_target_properties(_ops_dropout PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  PREFIX "" SUFFIX ".pyd"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
)
if (WIN32)
  set_target_properties(_ops_dropout PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
  )
endif()
apply_win_warnings(_ops_dropout)

# =========================
# Python 모듈 (_ops_conv2d) — 독립 바인딩
#  - conv2d가 im2col+gemm을 쓴다면 ge2_regemm_standalone 재사용
# =========================
pybind11_add_module(_ops_conv2d MODULE
  src/bindings/conv2d_pybind.cpp
  backends/cuda/ops/conv2d/launcher.cu
  backends/cuda/ops/conv2d/kernels.cu
)
target_compile_definitions(_ops_conv2d PRIVATE BUILD_STANDALONE_OPS=1)
target_include_directories(_ops_conv2d PRIVATE
  ${CMAKE_SOURCE_DIR}
  backends/cuda/ops/_common/shim
  backends/cuda/ops/conv2d
  backends/cuda/ops/gemm/include
  backends/cuda/ops/gemm
)
target_link_libraries(_ops_conv2d PRIVATE
  $<TARGET_OBJECTS:ge2_regemm_standalone>  # ← GEMM 재사용
  CUDA::cudart
  CUDA::cublas                              # im2col+gemm이라면 필요
  # CUDA::cudnn                              # cudnn 경로 사용 시
)
set_target_properties(_ops_conv2d PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  PREFIX "" SUFFIX ".pyd"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
)
if (WIN32)
  set_target_properties(_ops_conv2d PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
  )
endif()
apply_win_warnings(_ops_conv2d)

# =========================
# 상태 출력
# =========================
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "PROJECT_INCLUDE_DIR: ${PROJECT_INCLUDE_DIR}")
