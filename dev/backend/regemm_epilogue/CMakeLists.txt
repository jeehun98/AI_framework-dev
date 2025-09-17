cmake_minimum_required(VERSION 3.24)
project(regemm_epilogue LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# --- 언어 표준/아키 ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
# 필요시 다중 아키: 80-real 86 89 90 등
set(CMAKE_CUDA_ARCHITECTURES 86)

# --- 포함 경로 ---
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- 라이브러리(정적) ---
#  ✅ forward(타일/스모크) + EX 경로 + backward + 런처 포함
add_library(regemm_epilogue STATIC
  src/regemm_gemm_bias_act.cu
  src/regemm_backward.cu       # NEW: backward(EX)
  # src/launcher.cu              # NEW: EX 라우팅
)

# --- 컴파일 옵션 ---
target_compile_options(regemm_epilogue PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xptxas=-v>
)

# --- 링크 ---
#  ✅ cudart + cublas 필요 (backward에서 cublasSgemm/Sgeam 사용)
target_link_libraries(regemm_epilogue PRIVATE
  CUDA::cudart
  CUDA::cublas
)

# --- 벤치마크 ---
add_executable(bench_gemm benchmarks/bench_gemm.cu)
target_link_libraries(bench_gemm PRIVATE regemm_epilogue CUDA::cudart)
target_include_directories(bench_gemm PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

add_executable(bench_activations benchmarks/bench_activations.cu)
target_link_libraries(bench_activations PRIVATE regemm_epilogue CUDA::cudart)
target_include_directories(bench_activations PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# --- 테스트 ---
add_executable(test_backward tests/test_backward.cpp)
target_link_libraries(test_backward PRIVATE regemm_epilogue CUDA::cudart)
target_include_directories(test_backward PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
add_test(NAME test_backward COMMAND test_backward)

# --- CTest ---
enable_testing()
add_test(NAME test_basic COMMAND test_basic)
