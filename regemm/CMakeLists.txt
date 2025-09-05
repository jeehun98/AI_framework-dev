cmake_minimum_required(VERSION 3.26)
project(regemm LANGUAGES CXX CUDA)

# 맨 위나 project() 다음에
include(CTest)         # BUILD_TESTING 옵션 제공
enable_testing()       # ctest 사용 가능하게

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 아키 기본값
if (NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
endif()

# ★ CUDA 런타임 패키지 찾기
find_package(CUDAToolkit REQUIRED)

add_library(regemm STATIC
  src/gemm_bias_act_smoke.cu
  src/gemm_bias_act_tiled.cu
  src/launcher.cu
)
target_include_directories(regemm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 최적화 플래그

target_compile_options(regemm PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v>
)


# ★ regemm 사용자가 cudart 포함/링크를 자동으로 상속받도록 PUBLIC로 연결
target_link_libraries(regemm PUBLIC CUDA::cudart)

add_subdirectory(tests)

add_executable(bench_regemm benchmarks/bench_gemm.cu)
target_link_libraries(bench_regemm PRIVATE regemm)
