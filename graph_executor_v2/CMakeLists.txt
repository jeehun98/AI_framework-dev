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

if (MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

if (NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

option(BUILD_PYTHON "Build Python extension modules (standalone ops)" ON)

# =========================
# Python / pybind11 / CUDA Toolkit
# =========================
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)  # CUDA::cudart, CUDA::cublas 등

# =========================
# include 경로
# =========================
set(PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(SHIM_DIR ${CMAKE_SOURCE_DIR}/backends/cuda/ops/_common/shim)

# =========================
# 공통 경고/옵션 (/utf-8 포함)
# =========================
function(apply_win_warnings tgt)
  if (MSVC)
    target_compile_options(${tgt} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/W4 /Zc:__cplusplus /bigobj /permissive- /EHsc /utf-8>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/permissive->
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
    target_compile_definitions(${tgt} PRIVATE NOMINMAX)
  else()
    target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
  endif()
endfunction()

# =========================
# NVTX 토글 & 탐지 (미발견 시 자동 OFF)
#  - 헤더는 AI_USE_NVTX 매크로를 기대하므로 동일하게 정의
# =========================
option(USE_NVTX "Enable NVTX ranges" ON)
if (USE_NVTX)
  if (NOT DEFINED NVTX_ROOT)
    if (WIN32)
      set(NVTX_ROOT "C:/Program Files/NVIDIA Corporation/NVToolsExt")
    else()
      set(NVTX_ROOT "/usr/local/NVToolsExt")
    endif()
  endif()
  find_path(NVTX_INCLUDE_DIR nvToolsExt.h
            HINTS "${NVTX_ROOT}/include" "$ENV{CUDA_PATH}/include")
  if (WIN32)
    find_library(NVTOOLSEXT_LIB NAMES nvToolsExt64_1 nvToolsExt
                 HINTS "${NVTX_ROOT}/lib/x64" "$ENV{CUDA_PATH}/lib/x64" "$ENV{CUDA_PATH}/lib")
  else()
    find_library(NVTOOLSEXT_LIB NAMES nvToolsExt
                 HINTS "${NVTX_ROOT}/lib64" "${NVTX_ROOT}/lib" "$ENV{CUDA_PATH}/lib64" "$ENV{CUDA_PATH}/lib")
  endif()
  if (NOT NVTX_INCLUDE_DIR OR NOT NVTOOLSEXT_LIB)
    message(WARNING "NVTX not fully found (include='${NVTX_INCLUDE_DIR}', lib='${NVTOOLSEXT_LIB}'). Disabling USE_NVTX.")
    set(USE_NVTX OFF)
  endif()
endif()

# =========================
# 런타임 DLL 복사 (Windows)
# =========================
function(copy_runtime_dlls tgt outdir)
  if (WIN32)
    add_custom_command(TARGET ${tgt} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory "${outdir}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "$ENV{CUDA_PATH}/bin/cudart64_12.dll" "${outdir}" || ${CMAKE_COMMAND} -E echo "cudart64_12.dll not found; skip"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "$ENV{CUDA_PATH}/bin/cublas64_12.dll" "${outdir}" || ${CMAKE_COMMAND} -E echo "cublas64_12.dll not found; skip"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "$ENV{CUDA_PATH}/bin/cublasLt64_12.dll" "${outdir}" || ${CMAKE_COMMAND} -E echo "cublasLt64_12.dll not found; skip"
      $<$<BOOL:${USE_NVTX}>:${CMAKE_COMMAND}> -E copy_if_different "$ENV{CUDA_PATH}/bin/nvToolsExt64_1.dll" "${outdir}"
    )
  endif()
endfunction()

# =========================
# ai_shim (헤더 온리) 인터페이스 타깃
#  - 분리된 shim 헤더 경로를 중앙에서 제공
#  - NVTX 사용 시 AI_USE_NVTX 정의/링크 제공
# =========================
add_library(ai_shim INTERFACE)
target_include_directories(ai_shim INTERFACE
  ${PROJECT_INCLUDE_DIR}
  ${SHIM_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_features(ai_shim INTERFACE cxx_std_17)

if (USE_NVTX)
  target_compile_definitions(ai_shim INTERFACE AI_USE_NVTX=1)
  target_include_directories(ai_shim INTERFACE ${NVTX_INCLUDE_DIR})
  target_link_libraries(ai_shim INTERFACE ${NVTOOLSEXT_LIB})
endif()

# =========================
# Arena OBJECT (Capture-Safe Allocator)
#  - _ops_memory 모듈에서만 사용
# =========================
add_library(gev2_arena OBJECT
  src/executor/mem_planner.cpp
)
target_include_directories(gev2_arena PUBLIC
  ${CMAKE_SOURCE_DIR}
  ${PROJECT_INCLUDE_DIR}
)
target_include_directories(gev2_arena PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
apply_win_warnings(gev2_arena)

# =========================
# GEMM + Epilogue 오브젝트 (Standalone 전용)
# =========================
set(GE2_REGEMM_SOURCES
  backends/cuda/ops/gemm/launcher.cu
  backends/cuda/ops/gemm/kernels/regemm_gemm_bias_act.cu
  backends/cuda/ops/gemm/kernels/regemm_backward.cu
)
set(EPILOGUE_SRCS
  backends/cuda/ops/epilogue/launcher.cu
  backends/cuda/ops/epilogue/kernels.cu
)
add_library(ge2_regemm_standalone OBJECT ${GE2_REGEMM_SOURCES} ${EPILOGUE_SRCS})
set_target_properties(ge2_regemm_standalone PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(ge2_regemm_standalone PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${SHIM_DIR}
  backends/cuda/ops/gemm/include
  backends/cuda/ops/epilogue
)
target_compile_definitions(ge2_regemm_standalone PRIVATE BUILD_STANDALONE_OPS=1)
apply_win_warnings(ge2_regemm_standalone)
if (USE_NVTX)
  target_include_directories(ge2_regemm_standalone PRIVATE ${NVTX_INCLUDE_DIR})
endif()

# =========================
# 공통 함수: Py 모듈 생성 헬퍼
# =========================
function(make_ops_module name)
  # ARGN: extra sources
  pybind11_add_module(${name} MODULE ${ARGN})
  target_link_libraries(${name} PRIVATE ai_shim CUDA::cudart)
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR} ${SHIM_DIR})
  target_compile_definitions(${name} PRIVATE BUILD_STANDALONE_OPS=1)
  set_target_properties(${name} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    PREFIX "" SUFFIX ".pyd"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
  )
  if (WIN32)
    set_target_properties(${name} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
    )
  endif()
  apply_win_warnings(${name})
endfunction()

# =========================
# _ops_common (선행 로더)
# =========================
pybind11_add_module(_ops_common MODULE
  src/bindings/ops_common_pybind.cpp
)
target_link_libraries(_ops_common PRIVATE ai_shim)
target_include_directories(_ops_common PRIVATE ${PROJECT_INCLUDE_DIR} ${CMAKE_SOURCE_DIR})
set_target_properties(_ops_common PROPERTIES
  PREFIX "" SUFFIX ".pyd"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
)
if (WIN32)
  set_target_properties(_ops_common PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
  )
endif()
apply_win_warnings(_ops_common)

# =========================
# 개별 Ops 모듈들 (Standalone)
# =========================
# GEMM
make_ops_module(_ops_gemm
  src/bindings/gemm_pybind.cpp
)
target_sources(_ops_gemm PRIVATE $<TARGET_OBJECTS:ge2_regemm_standalone>)
target_link_libraries(_ops_gemm PRIVATE CUDA::cublas)
target_include_directories(_ops_gemm PRIVATE
  backends/cuda/ops/gemm/include
  backends/cuda/ops/epilogue
)

# conv2d
make_ops_module(_ops_conv2d
  src/bindings/conv2d_pybind.cpp
  backends/cuda/ops/conv2d/launcher.cu
  backends/cuda/ops/conv2d/kernels.cu
)
target_sources(_ops_conv2d PRIVATE $<TARGET_OBJECTS:ge2_regemm_standalone>)
target_link_libraries(_ops_conv2d PRIVATE CUDA::cublas)
target_include_directories(_ops_conv2d PRIVATE
  backends/cuda/ops/conv2d
  backends/cuda/ops/gemm/include
  backends/cuda/ops/epilogue
)

# memory (Arena 포함)
make_ops_module(_ops_memory
  src/bindings/memory_pybind.cpp
  backends/cuda/ops/memory/launcher.cu
  backends/cuda/ops/memory/kernels.cu
)
target_sources(_ops_memory PRIVATE $<TARGET_OBJECTS:gev2_arena>)

# dropout
make_ops_module(_ops_dropout
  src/bindings/dropout_pybind.cpp
  backends/cuda/ops/dropout/launcher.cu
  backends/cuda/ops/dropout/kernels.cu
)

# rmsnorm
make_ops_module(_ops_rmsnorm
  src/bindings/rmsnorm_pybind.cpp
  backends/cuda/ops/rmsnorm/launcher.cu
  backends/cuda/ops/rmsnorm/kernels.cu
)

# cross_entropy
make_ops_module(_ops_cross_entropy
  src/bindings/cross_entropy_pybind.cpp
  backends/cuda/ops/cross_entropy/launcher.cu
  backends/cuda/ops/cross_entropy/kernels.cu
)

# pad
make_ops_module(_ops_pad
  src/bindings/pad_pybind.cpp
  backends/cuda/ops/pad/launcher.cu
)

# pool2d
make_ops_module(_ops_pool2d
  src/bindings/pool2d_pybind.cpp
  backends/cuda/ops/pool2d/launcher.cu
  backends/cuda/ops/pool2d/kernels.cu
)

# rnn (원하면 잠시 주석 가능)
make_ops_module(_ops_rnn
  src/bindings/rnn_pybind.cpp
  backends/cuda/ops/rnn/launcher.cu
  backends/cuda/ops/rnn/kernels.cu
)
target_sources(_ops_rnn PRIVATE $<TARGET_OBJECTS:ge2_regemm_standalone>)
target_link_libraries(_ops_rnn PRIVATE CUDA::cublas)
target_include_directories(_ops_rnn PRIVATE
  backends/cuda/ops/gemm/include
  backends/cuda/ops/epilogue
  backends/cuda/ops/rnn
)

# optimizer
make_ops_module(_ops_optimizer
  src/bindings/optimizer_pybind.cpp
  backends/cuda/ops/optimizer/launcher.cu
  backends/cuda/ops/optimizer/kernels.cu
)

# softmax
make_ops_module(_ops_softmax
  src/bindings/softmax_pybind.cpp
  backends/cuda/ops/softmax/launcher.cu
  backends/cuda/ops/softmax/kernels.cu
)

# batchnorm
make_ops_module(_ops_batchnorm
  src/bindings/batchnorm_pybind.cpp
  backends/cuda/ops/batchnorm/launcher.cu
  backends/cuda/ops/batchnorm/kernels.cu
)

# embedding
make_ops_module(_ops_embedding
  src/bindings/embedding_pybind.cpp
  backends/cuda/ops/embedding/launcher.cu
  backends/cuda/ops/embedding/kernels.cu
)

# layernorm
make_ops_module(_ops_layernorm
  src/bindings/layernorm_pybind.cpp
  backends/cuda/ops/layernorm/launcher.cu
  backends/cuda/ops/layernorm/kernels.cu
)

# concat
make_ops_module(_ops_concat
  src/bindings/concat_pybind.cpp
  backends/cuda/ops/concat/launcher.cu
  backends/cuda/ops/concat/kernels.cu
)

# slice
make_ops_module(_ops_slice
  src/bindings/slice_pybind.cpp
  backends/cuda/ops/slice/launcher.cu
  backends/cuda/ops/slice/kernels.cu
)

# view
make_ops_module(_ops_view
  src/bindings/view_pybind.cpp
  backends/cuda/ops/view/launcher.cu
  backends/cuda/ops/view/kernels.cu
)

# _ops_common 먼저 빌드되도록 의존 추가
set(_ops_modules
  _ops_gemm _ops_conv2d _ops_pool2d _ops_pad _ops_dropout _ops_rnn
  _ops_optimizer _ops_memory _ops_batchnorm _ops_embedding _ops_layernorm
  _ops_softmax _ops_cross_entropy _ops_rmsnorm _ops_concat _ops_slice _ops_view
)
foreach(mod IN LISTS _ops_modules)
  if (TARGET ${mod})
    add_dependencies(${mod} _ops_common)
  endif()
endforeach()

# =========================
# DLL 복사 (Windows)
# =========================
if (WIN32)
  set(_PY_OUT_OPS "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops")
  foreach(mod IN LISTS _ops_modules)
    if (TARGET ${mod})
      copy_runtime_dlls(${mod} "${_PY_OUT_OPS}")
    endif()
  endforeach()
  # _ops_common도 복사
  if (TARGET _ops_common)
    copy_runtime_dlls(_ops_common "${_PY_OUT_OPS}")
  endif()
endif()

# =========================
# 상태 출력
# =========================
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Standalone ops only: ENABLED")
if (USE_NVTX)
  message(STATUS "NVTX: enabled, include=${NVTX_INCLUDE_DIR}, lib=${NVTOOLSEXT_LIB}")
else()
  message(STATUS "NVTX: disabled")
endif()
