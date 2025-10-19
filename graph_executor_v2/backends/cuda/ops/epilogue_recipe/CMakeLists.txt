cmake_minimum_required(VERSION 3.23)

project(epilogue LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GPU 아키텍처 (수정 가능)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# 정적 라이브러리: 모든 CUDA 컴파일 단위를 한 타깃에 모음
add_library(epi STATIC
  launcher/epilogue_launcher_entry.cu
  launcher/epilogue_launcher_fp32.cu
  launcher/epilogue_launcher_fp16.cu

  kernels/generic/epilogue_generic_f32.cu
  kernels/generic/epilogue_generic_f16.cu
  kernels/fused/ep_f32_relu_bias.cu
  kernels/fused/ep_f32_relu_nobias.cu
  kernels/fused/ep_f16_relu_bias.cu
  kernels/fused/ep_f16_relu_nobias.cu

  kernels/fused/ep_f32_gelu_bias.cu
  kernels/fused/ep_f32_gelu_nobias.cu
  kernels/fused/ep_f16_gelu_bias.cu
  kernels/fused/ep_f16_gelu_nobias.cu

  kernels/fused/ep_f32_ffn_gelu_drop_resid.cu
  kernels/fused/ep_f16_ffn_gelu_drop_resid.cu

  kernels/fused/ep_f32_gru3.cu
  kernels/fused/ep_f16_gru3.cu


)

target_include_directories(epi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_package(CUDAToolkit REQUIRED)
target_link_libraries(epi PUBLIC CUDA::cudart)

set_target_properties(epi PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  POSITION_INDEPENDENT_CODE ON
)

target_compile_options(epi PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
)

# 테스트
add_executable(epi_test tests/test_relu_bias_fp32.cpp)
target_link_libraries(epi_test PRIVATE epi CUDA::cudart)
set_target_properties(epi_test PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(epi_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
)
