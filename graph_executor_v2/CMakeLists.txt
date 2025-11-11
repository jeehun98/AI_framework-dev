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
# 공통 경고/옵션 헬퍼  (/utf-8 추가)
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
# NVTX 토글 & 탐지 (못 찾으면 자동 OFF)
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
    message(WARNING "NVTX not fully found (include='${NVTX_INCLUDE_DIR}', lib='${NVTOOLSEXT_LIB}'). Disabling USE_NVTX to avoid runtime DLL errors.")
    set(USE_NVTX OFF)
  else()
    add_definitions(-DUSE_NVTX)
  endif()
endif()

# =========================
# 런타임 DLL 복사 헬퍼 (Windows)
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
# Arena OBJECT 라이브러리 (Capture-Safe Allocator)
#  - core와 _ops_memory 양쪽에서 재사용
# =========================
add_library(gev2_arena OBJECT
  src/executor/mem_planner.cpp
)
target_include_directories(gev2_arena PUBLIC
  ${CMAKE_SOURCE_DIR}
  ${PROJECT_INCLUDE_DIR}
)

target_include_directories(gev2_arena PUBLIC ${CUDAToolkit_INCLUDE_DIRS})  # 방법 B

apply_win_warnings(gev2_arena)

# =========================
# ai_core (디스패치/실행기/ops)
#  - mem_planner.cpp는 직접 빼고, OBJECT 주입으로 포함
# =========================
add_library(ai_core STATIC
  src/dispatch/registry.cpp
  src/dispatch/selector_rules.cpp
  src/executor/scheduler.cpp
  # src/executor/mem_planner.cpp  # <- 제거 (OBJECT로 분리)
  src/executor/autograd_engine.cpp
  src/ops/gemm.cpp
  src/ops/rmsnorm.cpp
  src/ops/softmax.cpp
  src/ops/layernorm.cpp
  src/ops/cross_entropy.cpp
  src/ops/dropout.cpp
  src/ops/sdpa.cpp

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
  src/ops/rnn.cpp
)
# Arena OBJECT 주입
target_sources(ai_core PRIVATE $<TARGET_OBJECTS:gev2_arena>)

target_include_directories(ai_core
  PUBLIC
    ${PROJECT_INCLUDE_DIR}
  PRIVATE
    ${CMAKE_SOURCE_DIR}
)
apply_win_warnings(ai_core)
if (USE_NVTX)
  target_include_directories(ai_core PRIVATE ${NVTX_INCLUDE_DIR})
  target_link_libraries(ai_core PRIVATE ${NVTOOLSEXT_LIB})
endif()

# =========================
# GEMM + Epilogue 공용 오브젝트 라이브러리 (소스 공통)
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

# (1) 통합 백엔드용
add_library(ge2_regemm_core OBJECT ${GE2_REGEMM_SOURCES} ${EPILOGUE_SRCS})
set_property(TARGET ge2_regemm_core PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(ge2_regemm_core
  PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${PROJECT_INCLUDE_DIR}
    backends/cuda/ops/gemm/include
    backends/cuda/ops/epilogue
)
apply_win_warnings(ge2_regemm_core)
if (USE_NVTX)
  target_include_directories(ge2_regemm_core PRIVATE ${NVTX_INCLUDE_DIR})
endif()

# (2) 독립 모듈용
add_library(ge2_regemm_standalone OBJECT ${GE2_REGEMM_SOURCES} ${EPILOGUE_SRCS})
set_target_properties(ge2_regemm_standalone PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(ge2_regemm_standalone
  PRIVATE
    ${CMAKE_SOURCE_DIR}
    backends/cuda/ops/_common/shim
    backends/cuda/ops/gemm/include
    backends/cuda/ops/epilogue
)
target_compile_definitions(ge2_regemm_standalone PRIVATE BUILD_STANDALONE_OPS=1)
apply_win_warnings(ge2_regemm_standalone)
if (USE_NVTX)
  target_include_directories(ge2_regemm_standalone PRIVATE ${NVTX_INCLUDE_DIR})
endif()

# =========================
# ai_backend_cuda (CUDA 백엔드 + 모든 CUDA ops)
# =========================
add_library(ai_backend_cuda STATIC
  src/ops/register_ops.cpp

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

  backends/cuda/ops/concat/kernels.cu
  backends/cuda/ops/concat/launcher.cu

  backends/cuda/ops/slice/kernels.cu
  backends/cuda/ops/slice/launcher.cu

  backends/cuda/ops/pad/launcher.cu

  backends/cuda/ops/memory/launcher.cu
  backends/cuda/ops/memory/kernels.cu

  backends/cuda/ops/upsample2d/launcher.cu

  backends/cuda/ops/view/kernels.cu
  backends/cuda/ops/view/launcher.cu

  backends/cuda/ops/indexing/kernels.cu
  backends/cuda/ops/indexing/launcher.cu

  backends/cuda/ops/rnn/kernels.cu
  backends/cuda/ops/rnn/launcher.cu

  backends/cuda/ops/optimizer/kernels.cu
  backends/cuda/ops/optimizer/launcher.cu

  backends/cuda/ops/batchnorm/kernels.cu
  backends/cuda/ops/batchnorm/launcher.cu

  backends/cuda/ops/embedding/kernels.cu
  backends/cuda/ops/embedding/launcher.cu

  $<TARGET_OBJECTS:ge2_regemm_core>
)
set_target_properties(ai_backend_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_include_directories(ai_backend_cuda
  PUBLIC
    backends/cuda/ops/gemm/detail
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
if (USE_NVTX)
  target_include_directories(ai_backend_cuda PRIVATE ${NVTX_INCLUDE_DIR})
  target_link_libraries(ai_backend_cuda PRIVATE ${NVTOOLSEXT_LIB})
endif()
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
# Python 모듈 (_ops_common)
# =========================
pybind11_add_module(_ops_common MODULE
  src/bindings/ops_common_pybind.cpp
)
target_include_directories(_ops_common PRIVATE
  ${PROJECT_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}
)
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
# Python 모듈 (_ops_gemm)
# =========================
pybind11_add_module(_ops_gemm MODULE
  src/bindings/gemm_pybind.cpp
)
target_sources(_ops_gemm PRIVATE
  $<TARGET_OBJECTS:ge2_regemm_standalone>
)
target_link_libraries(_ops_gemm PRIVATE
  CUDA::cudart
  CUDA::cublas
)
if (USE_NVTX)
  target_include_directories(_ops_gemm PRIVATE ${NVTX_INCLUDE_DIR})
  target_link_libraries(_ops_gemm PRIVATE ${NVTOOLSEXT_LIB})
endif()
target_include_directories(_ops_gemm PRIVATE
  ${CMAKE_SOURCE_DIR}
  backends/cuda/ops/_common/shim
  backends/cuda/ops/gemm/include
  backends/cuda/ops/epilogue
)
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
# Python 모듈 (_ops_conv2d)
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
  backends/cuda/ops/epilogue
)
target_sources(_ops_conv2d PRIVATE
  $<TARGET_OBJECTS:ge2_regemm_standalone>
)
target_link_libraries(_ops_conv2d PRIVATE
  CUDA::cudart
  CUDA::cublas
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
# Python 모듈 (_ops_memory)
# =========================
pybind11_add_module(_ops_memory MODULE
  src/bindings/memory_pybind.cpp
  backends/cuda/ops/memory/launcher.cu
  backends/cuda/ops/memory/kernels.cu
)
target_compile_definitions(_ops_memory PRIVATE BUILD_STANDALONE_OPS=1)
target_include_directories(_ops_memory PRIVATE
  ${CMAKE_SOURCE_DIR}
  backends/cuda/ops/_common/shim
  backends/cuda/ops/memory
)
# ⬇️ Arena OBJECT를 모듈에 포함 (핵심 변경)
target_sources(_ops_memory PRIVATE $<TARGET_OBJECTS:gev2_arena>)
target_link_libraries(_ops_memory PRIVATE CUDA::cudart)
set_target_properties(_ops_memory PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  PREFIX "" SUFFIX ".pyd"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
)
if (WIN32)
  set_target_properties(_ops_memory PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
  )
endif()
apply_win_warnings(_ops_memory)

# =========================
# Python 모듈 (_ops_dropout)
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
# Python 모듈 (_ops_rmsnorm)
# =========================
pybind11_add_module(_ops_rmsnorm MODULE
  src/bindings/rmsnorm_pybind.cpp
  backends/cuda/ops/rmsnorm/launcher.cu
  backends/cuda/ops/rmsnorm/kernels.cu
)
target_compile_definitions(_ops_rmsnorm PRIVATE BUILD_STANDALONE_OPS=1)
target_include_directories(_ops_rmsnorm PRIVATE
  ${CMAKE_SOURCE_DIR}
  backends/cuda/ops/_common/shim
  backends/cuda/ops/rmsnorm
)
target_link_libraries(_ops_rmsnorm PRIVATE CUDA::cudart)
set_target_properties(_ops_rmsnorm PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  PREFIX "" SUFFIX ".pyd"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
)
if (WIN32)
  set_target_properties(_ops_rmsnorm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
  )
endif()
apply_win_warnings(_ops_rmsnorm)

# =========================
# Python 모듈 (_ops_cross_entropy)
# =========================
pybind11_add_module(_ops_cross_entropy MODULE
  src/bindings/cross_entropy_pybind.cpp
  backends/cuda/ops/cross_entropy/launcher.cu
  backends/cuda/ops/cross_entropy/kernels.cu
)
target_compile_definitions(_ops_cross_entropy PRIVATE BUILD_STANDALONE_OPS=1)
target_include_directories(_ops_cross_entropy PRIVATE
  ${CMAKE_SOURCE_DIR}
  backends/cuda/ops/_common/shim
  backends/cuda/ops/cross_entropy
)
target_link_libraries(_ops_cross_entropy PRIVATE CUDA::cudart)
set_target_properties(_ops_cross_entropy PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  PREFIX "" SUFFIX ".pyd"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
)
if (WIN32)
  set_target_properties(_ops_cross_entropy PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
  )
endif()
apply_win_warnings(_ops_cross_entropy)

# =========================
# Python 모듈 (_ops_pad)
# =========================
pybind11_add_module(_ops_pad MODULE
  src/bindings/pad_pybind.cpp
  backends/cuda/ops/pad/launcher.cu
)
target_compile_definitions(_ops_pad PRIVATE BUILD_STANDALONE_OPS=1)
target_include_directories(_ops_pad PRIVATE
  ${CMAKE_SOURCE_DIR}
  backends/cuda/ops/_common/shim
  backends/cuda/ops/pad
)
target_link_libraries(_ops_pad PRIVATE CUDA::cudart)
set_target_properties(_ops_pad PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  PREFIX "" SUFFIX ".pyd"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
)
if (WIN32)
  set_target_properties(_ops_pad PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
  )
endif()
apply_win_warnings(_ops_pad)

# =========================
# Python 모듈 (_ops_pool2d)
# =========================
pybind11_add_module(_ops_pool2d MODULE
  src/bindings/pool2d_pybind.cpp
  backends/cuda/ops/pool2d/launcher.cu
  backends/cuda/ops/pool2d/kernels.cu
)
target_compile_definitions(_ops_pool2d PRIVATE BUILD_STANDALONE_OPS=1)
target_include_directories(_ops_pool2d PRIVATE
  ${CMAKE_SOURCE_DIR}
  backends/cuda/ops/_common/shim
  backends/cuda/ops/pool2d
)
target_link_libraries(_ops_pool2d PRIVATE CUDA::cudart)
set_target_properties(_ops_pool2d PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  PREFIX "" SUFFIX ".pyd"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
)
if (WIN32)
  set_target_properties(_ops_pool2d PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
  )
endif()
apply_win_warnings(_ops_pool2d)

# =========================
# Python 모듈 (_ops_rnn)
# =========================
pybind11_add_module(_ops_rnn MODULE
  src/bindings/rnn_pybind.cpp
  backends/cuda/ops/rnn/launcher.cu
  backends/cuda/ops/rnn/kernels.cu
)
target_compile_definitions(_ops_rnn PRIVATE BUILD_STANDALONE_OPS=1)
target_include_directories(_ops_rnn PRIVATE
  ${CMAKE_SOURCE_DIR}
  backends/cuda/ops/_common/shim
  backends/cuda/ops/gemm/include
  backends/cuda/ops/rnn
  backends/cuda/ops/epilogue
)

# ↓ regemm 오브젝트(안에 NVTX 호출 있음)를 가져오므로 NVTX 링크 필요
target_sources(_ops_rnn PRIVATE
  $<TARGET_OBJECTS:ge2_regemm_standalone>
)
target_link_libraries(_ops_rnn PRIVATE
  CUDA::cudart
  CUDA::cublas
)

# ★ 여기 추가
if (USE_NVTX)
  target_include_directories(_ops_rnn PRIVATE ${NVTX_INCLUDE_DIR})
  target_link_libraries(_ops_rnn PRIVATE ${NVTOOLSEXT_LIB})
endif()

set_target_properties(_ops_rnn PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  PREFIX "" SUFFIX ".pyd"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
)



# =========================
# Python 모듈 (_ops_optimizer)
# =========================
pybind11_add_module(_ops_optimizer MODULE
  src/bindings/optimizer_pybind.cpp
  backends/cuda/ops/optimizer/launcher.cu
  backends/cuda/ops/optimizer/kernels.cu
)
target_compile_definitions(_ops_optimizer PRIVATE BUILD_STANDALONE_OPS=1)
target_include_directories(_ops_optimizer PRIVATE
  ${CMAKE_SOURCE_DIR}
  backends/cuda/ops/_common/shim
  backends/cuda/ops/optimizer
)
target_link_libraries(_ops_optimizer PRIVATE CUDA::cudart)
set_target_properties(_ops_optimizer PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  PREFIX "" SUFFIX ".pyd"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
)
if (WIN32)
  set_target_properties(_ops_optimizer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
  )
endif()
apply_win_warnings(_ops_optimizer)

# =========================
# Python 모듈 (_ops_softmax)
# =========================
pybind11_add_module(_ops_softmax MODULE
  src/bindings/softmax_pybind.cpp
  backends/cuda/ops/softmax/launcher.cu
  backends/cuda/ops/softmax/kernels.cu
)
target_compile_definitions(_ops_softmax PRIVATE BUILD_STANDALONE_OPS=1)
target_include_directories(_ops_softmax PRIVATE
  ${CMAKE_SOURCE_DIR}
  backends/cuda/ops/_common/shim
  backends/cuda/ops/softmax
)
target_link_libraries(_ops_softmax PRIVATE CUDA::cudart)
set_target_properties(_ops_softmax PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  PREFIX "" SUFFIX ".pyd"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
)
if (WIN32)
  set_target_properties(_ops_softmax PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
  )
endif()
apply_win_warnings(_ops_softmax)

# =========================
# Python 모듈 (_ops_batchnorm)
# =========================
pybind11_add_module(_ops_batchnorm MODULE
  src/bindings/batchnorm_pybind.cpp
  backends/cuda/ops/batchnorm/launcher.cu
  backends/cuda/ops/batchnorm/kernels.cu
)
target_compile_definitions(_ops_batchnorm PRIVATE BUILD_STANDALONE_OPS=1)
target_include_directories(_ops_batchnorm PRIVATE
  ${CMAKE_SOURCE_DIR}
  backends/cuda/ops/_common/shim
  backends/cuda/ops/batchnorm
)
target_link_libraries(_ops_batchnorm PRIVATE
  CUDA::cudart
)
set_target_properties(_ops_batchnorm PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  PREFIX "" SUFFIX ".pyd"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
)
if (WIN32)
  set_target_properties(_ops_batchnorm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
  )
endif()
apply_win_warnings(_ops_batchnorm)

# =========================
# Python 모듈 (_ops_embedding)
# =========================
pybind11_add_module(_ops_embedding MODULE
  src/bindings/embedding_pybind.cpp
  backends/cuda/ops/embedding/launcher.cu
  backends/cuda/ops/embedding/kernels.cu
)
target_compile_definitions(_ops_embedding PRIVATE BUILD_STANDALONE_OPS=1)
target_include_directories(_ops_embedding PRIVATE
  ${CMAKE_SOURCE_DIR}
  backends/cuda/ops/_common/shim
  backends/cuda/ops/embedding
)
target_link_libraries(_ops_embedding PRIVATE CUDA::cudart)
set_target_properties(_ops_embedding PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  PREFIX "" SUFFIX ".pyd"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
)
if (WIN32)
  set_target_properties(_ops_embedding PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
  )
endif()
apply_win_warnings(_ops_embedding)

# =========================
# Python 모듈 (_ops_layernorm)
# =========================
pybind11_add_module(_ops_layernorm MODULE
  src/bindings/layernorm_pybind.cpp
  backends/cuda/ops/layernorm/launcher.cu
  backends/cuda/ops/layernorm/kernels.cu
)
target_compile_definitions(_ops_layernorm PRIVATE BUILD_STANDALONE_OPS=1)
target_include_directories(_ops_layernorm PRIVATE
  ${CMAKE_SOURCE_DIR}
  backends/cuda/ops/_common/shim
  backends/cuda/ops/layernorm
)
target_link_libraries(_ops_layernorm PRIVATE CUDA::cudart)
set_target_properties(_ops_layernorm PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  PREFIX "" SUFFIX ".pyd"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
)
if (WIN32)
  set_target_properties(_ops_layernorm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
  )
endif()
apply_win_warnings(_ops_layernorm)

# =========================
# Python 모듈 (_ops_concat)
# =========================
pybind11_add_module(_ops_concat MODULE
  src/bindings/concat_pybind.cpp
  backends/cuda/ops/concat/launcher.cu
  backends/cuda/ops/concat/kernels.cu
)
target_compile_definitions(_ops_concat PRIVATE BUILD_STANDALONE_OPS=1)
target_include_directories(_ops_concat PRIVATE
  ${CMAKE_SOURCE_DIR}
  backends/cuda/ops/_common/shim
  backends/cuda/ops/concat
)
target_link_libraries(_ops_concat PRIVATE CUDA::cudart)
set_target_properties(_ops_concat PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  PREFIX "" SUFFIX ".pyd"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
)
if (WIN32)
  set_target_properties(_ops_concat PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
  )
endif()
apply_win_warnings(_ops_concat)

# =========================
# Python 모듈 (_ops_slice)
# =========================
pybind11_add_module(_ops_slice MODULE
  src/bindings/slice_pybind.cpp
  backends/cuda/ops/slice/launcher.cu
  backends/cuda/ops/slice/kernels.cu
)
target_compile_definitions(_ops_slice PRIVATE BUILD_STANDALONE_OPS=1)
target_include_directories(_ops_slice PRIVATE
  ${CMAKE_SOURCE_DIR}
  backends/cuda/ops/_common/shim
  backends/cuda/ops/slice
)
target_link_libraries(_ops_slice PRIVATE CUDA::cudart)
set_target_properties(_ops_slice PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  PREFIX "" SUFFIX ".pyd"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
)
if (WIN32)
  set_target_properties(_ops_slice PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
  )
endif()
apply_win_warnings(_ops_slice)

# =========================
# Python 모듈 (_ops_view)
# =========================
pybind11_add_module(_ops_view MODULE
  src/bindings/view_pybind.cpp
  backends/cuda/ops/view/launcher.cu
  backends/cuda/ops/view/kernels.cu
)
target_compile_definitions(_ops_view PRIVATE BUILD_STANDALONE_OPS=1)
target_include_directories(_ops_view PRIVATE
  ${CMAKE_SOURCE_DIR}
  backends/cuda/ops/_common/shim
  backends/cuda/ops/view
)
target_link_libraries(_ops_view PRIVATE CUDA::cudart)
set_target_properties(_ops_view PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  PREFIX "" SUFFIX ".pyd"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
)
if (WIN32)
  set_target_properties(_ops_view PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops"
  )
endif()
apply_win_warnings(_ops_view)

# =========================
# _ops_common 선행 의존성 일괄 추가
# =========================
set(_ops_modules
  _ops_gemm _ops_conv2d _ops_pool2d _ops_pad _ops_dropout _ops_rnn
  _ops_optimizer _ops_memory _ops_batchnorm _ops_embedding _ops_layernorm
  _ops_softmax _ops_cross_entropy _ops_rmsnorm
  _ops_concat _ops_slice _ops_view
)
foreach(mod IN LISTS _ops_modules)
  if (TARGET ${mod} AND TARGET _ops_common)
    add_dependencies(${mod} _ops_common)
  endif()
endforeach()

# =========================
# 필수 DLL 동반 복사 적용 (Windows)
# =========================
if (WIN32)
  set(_PY_OUT_CORE "${CMAKE_SOURCE_DIR}/python/graph_executor_v2")
  set(_PY_OUT_OPS  "${CMAKE_SOURCE_DIR}/python/graph_executor_v2/ops")

  if (TARGET _core)
    copy_runtime_dlls(_core "${_PY_OUT_CORE}")
  endif()

  foreach(mod IN LISTS _ops_modules)
    if (TARGET ${mod})
      copy_runtime_dlls(${mod} "${_PY_OUT_OPS}")
    endif()
  endforeach()
endif()

# =========================
# 상태 출력
# =========================
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "PROJECT_INCLUDE_DIR: ${PROJECT_INCLUDE_DIR}")
if (USE_NVTX)
  message(STATUS "NVTX: enabled, include=${NVTX_INCLUDE_DIR}, lib=${NVTOOLSEXT_LIB}")
else()
  message(STATUS "NVTX: disabled")
endif()
