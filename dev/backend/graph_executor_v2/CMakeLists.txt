cmake_minimum_required(VERSION 3.26)
project(graph_executor_v2 LANGUAGES CXX CUDA)

# Python / pybind11
find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG QUIET)
if (NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.12.0
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# CUDA
find_package(CUDAToolkit REQUIRED)
# Ampere 예: 86 (필요 시 ;로 여러 개 지정)
set(CMAKE_CUDA_ARCHITECTURES 86)

set(SRC
  src/bindings_min_api.cpp
  src/launch_table.cpp
  src/my_kernels.cu
)

pybind11_add_module(graph_executor_v2 MODULE ${SRC})

target_compile_features(graph_executor_v2 PRIVATE cxx_std_17)
set_target_properties(graph_executor_v2 PROPERTIES
  CUDA_STANDARD 17
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME "graph_executor_v2"
)

if (MSVC)
  target_compile_options(graph_executor_v2 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/bigobj;/W3;/EHsc;/O2;/MD>
  )
else()
  target_compile_options(graph_executor_v2 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-O2>
  )
endif()

# CUDA 호스트 옵션 전달
target_compile_options(graph_executor_v2 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc;-Xcompiler=/O2;-Xcompiler=/MD>
)

# 링크: cuBLASLt 포함
target_link_libraries(graph_executor_v2 PRIVATE
  CUDA::cudart
  CUDA::cublas
  CUDA::cublasLt
)

target_include_directories(graph_executor_v2 PRIVATE
  ${CUDAToolkit_INCLUDE_DIRS}
  include
)

install(TARGETS graph_executor_v2
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .
        ARCHIVE DESTINATION .)

install(FILES include/ge_v2_api.h DESTINATION include)
