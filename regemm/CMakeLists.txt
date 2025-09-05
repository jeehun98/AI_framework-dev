cmake_minimum_required(VERSION 3.26)
project(regemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
# 합리적 기본 아키; 필요시 -DCMAKE_CUDA_ARCHITECTURES="86;89;90"
if (NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
endif()

add_library(regemm STATIC
  src/gemm_bias_act_smoke.cu
  src/launcher.cu
)
target_include_directories(regemm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(regemm PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
)

add_subdirectory(tests)
add_executable(bench_regemm benchmarks/bench_gemm.cu)
target_link_libraries(bench_regemm PRIVATE regemm cuda)
