cmake_minimum_required(VERSION 3.24)
project(regemm_epilogue LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ✅ 단일 파일만 라이브러리에 포함
add_library(regemm_epilogue STATIC
  src/regemm_gemm_bias_act.cu
)

target_compile_options(regemm_epilogue PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: --use_fast_math -Xptxas=-v >
)

# ✅ cudart 링크
target_link_libraries(regemm_epilogue PRIVATE CUDA::cudart)

# Bench
add_executable(bench_gemm benchmarks/bench_gemm.cu)
target_link_libraries(bench_gemm PRIVATE regemm_epilogue CUDA::cudart)
target_include_directories(bench_gemm PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

add_executable(bench_activations benchmarks/bench_activations.cu)
target_link_libraries(bench_activations PRIVATE regemm_epilogue CUDA::cudart)
target_include_directories(bench_activations PRIVATE ${CUDAToolkit_INCLUDE_DIRS})


# Tests
add_executable(test_basic tests/test_basic.cpp)
target_link_libraries(test_basic PRIVATE regemm_epilogue CUDA::cudart)
target_include_directories(test_basic PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# ✅ CTest 등록
enable_testing()
add_test(NAME test_basic COMMAND test_basic)
