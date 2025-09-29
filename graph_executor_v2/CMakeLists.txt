cmake_minimum_required(VERSION 3.24)

project(graph_executor_v2 LANGUAGES CXX CUDA)

# =========================
# 기본 설정
# =========================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA 아키텍처 지정 (예: 80;86;89;90)
if (NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)  # 필요 시 -D로 오버라이드
endif()

option(BUILD_PYTHON "Build Python extension module (_core)" ON)

# =========================
# Python / pybind11 / CUDA Toolkit
# =========================
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)  # CUDA::cudart, CUDA::cublas 등 제공

# =========================
# include 경로 변수
# =========================
set(PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

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
  # (주의) 여기에는 CUDA 백엔드 등록/런처를 넣지 않습니다.
)

target_include_directories(ai_core
  PUBLIC
    ${PROJECT_INCLUDE_DIR}
  PRIVATE
    ${CMAKE_SOURCE_DIR}
)

if (MSVC)
  target_compile_options(ai_core PRIVATE
    /W4           # 경고 레벨
    /Zc:__cplusplus
    /bigobj
    /permissive-
    /EHsc
    /MD
  )
  target_compile_definitions(ai_core PRIVATE NOMINMAX)
else()
  target_compile_options(ai_core PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
endif()

# =========================
# GEMM 커널(Object 라이브러리; 백엔드에서만 결합)
# =========================
add_library(gemm_kernels_obj OBJECT
  backends/cuda/ops/gemm/kernels/regemm_gemm_bias_act.cu
  backends/cuda/ops/gemm/kernels/regemm_backward.cu
)

set_target_properties(gemm_kernels_obj PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(gemm_kernels_obj
  PRIVATE
    backends/cuda/ops/gemm/include     # "regemm/*.h"
    ${CMAKE_SOURCE_DIR}
    ${PROJECT_INCLUDE_DIR}
)

# =========================
# ai_backend_cuda (CUDA 백엔드 + 모든 CUDA ops)
# =========================
add_library(ai_backend_cuda STATIC
  src/ops/register_ops.cpp           # ← CUDA 백엔드의 op 등록 진입점

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
  backends/cuda/ops/slice/launcher.cu
  backends/cuda/ops/pad/launcher.cu

  backends/cuda/ops/memory/launcher.cu
  backends/cuda/ops/memory/kernels.cu

  backends/cuda/ops/upsample2d/launcher.cu
  backends/cuda/ops/view/launcher.cu

  backends/cuda/ops/indexing/kernels.cu
  backends/cuda/ops/indexing/launcher.cu

  $<TARGET_OBJECTS:gemm_kernels_obj>      # ← GEMM 커널 구현 결합 (중복 추가 금지)
)

set_target_properties(ai_backend_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(ai_backend_cuda
  PUBLIC
    backends/cuda/ops/gemm/include       # regemm/api.h 외부에서 필요 시
    backends/cuda/ops/_common/shim       # ai_shim.hpp (안전망)
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

if (MSVC)
  target_compile_options(ai_backend_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4 /Zc:__cplusplus /bigobj /permissive- /EHsc /MD>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
  )
  target_compile_definitions(ai_backend_cuda PRIVATE NOMINMAX)
else()
  target_compile_options(ai_backend_cuda PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
endif()

# =========================
# Python 모듈 (_core)
# =========================
if (BUILD_PYTHON)
  pybind11_add_module(_core MODULE
    src/bindings/py_api.cpp
  )
  target_link_libraries(_core PRIVATE ai_core ai_backend_cuda)

  target_include_directories(_core
    PRIVATE
      ${PROJECT_INCLUDE_DIR}
      ${CMAKE_SOURCE_DIR}
  )

  set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2"
  )
  if (WIN32)
    set_target_properties(_core PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2"
    )
  endif()

  if (MSVC)
    target_compile_options(_core PRIVATE /W4 /Zc:__cplusplus /bigobj /permissive- /EHsc /MD)
    target_compile_definitions(_core PRIVATE NOMINMAX)
  else()
    target_compile_options(_core PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
  endif()
endif()

# =========================
# Python 모듈 (_ops_gemm) — op 전용 독립 바인딩
# =========================
pybind11_add_module(_ops_gemm MODULE
  src/bindings/gemm_pybind.cpp

  # GEMM 런처/커널
  backends/cuda/ops/gemm/launcher.cu
  backends/cuda/ops/gemm/backward.cu
  backends/cuda/ops/gemm/kernels/regemm_gemm_bias_act.cu
  backends/cuda/ops/gemm/kernels/regemm_backward.cu
)

# 독립 빌드 플래그 (shim이 얇은 타입을 제공)
target_compile_definitions(_ops_gemm PRIVATE BUILD_STANDALONE_OPS=1)

# include 경로
target_include_directories(_ops_gemm PRIVATE
  ${CMAKE_SOURCE_DIR}                 # backends/... 상대 include 허용
  backends/cuda/ops/_common/shim      # ai_shim.hpp
  backends/cuda/ops/gemm/include      # regemm headers
)

# 링크는 CUDA 런타임/BLAS만
target_link_libraries(_ops_gemm PRIVATE
  CUDA::cudart
  CUDA::cublas
)

set_target_properties(_ops_gemm PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  OUTPUT_NAME "_ops_gemm"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
)

if (WIN32)
  set_target_properties(_ops_gemm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
  )
endif()

if (MSVC)
  target_compile_options(_ops_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4 /Zc:__cplusplus /bigobj /permissive- /EHsc /MD>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
  )
  target_compile_definitions(_ops_gemm PRIVATE NOMINMAX)
else()
  target_compile_options(_ops_gemm PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
endif()

# =========================
# 상태 출력
# =========================
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "PROJECT_INCLUDE_DIR: ${PROJECT_INCLUDE_DIR}")
