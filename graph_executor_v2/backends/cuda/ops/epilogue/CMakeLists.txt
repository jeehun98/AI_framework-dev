cmake_minimum_required(VERSION 3.23)

project(epilogue LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GPU에 맞게 수정 (Ampere=86, Ada=89 등)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# ---- 정적 라이브러리: 런처 + 커널 ----
add_library(epi STATIC
  launcher/epilogue_launcher.cu
  kernels/epilogue_kernel.cu
)

target_include_directories(epi PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(epi PUBLIC CUDA::cudart)

# CUDA: 분리 컴파일 + 디바이스 심볼 해결
set_target_properties(epi PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  POSITION_INDEPENDENT_CODE ON
)

# 인코딩: CXX=/utf-8, CUDA=-Xcompiler=/utf-8
target_compile_options(epi PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
)

# ---- 테스트 실행 파일 ----
add_executable(epi_test
  tests/test_epilogue_min.cpp
)

target_link_libraries(epi_test PRIVATE
  epi
  CUDA::cudart
)

# (안전) 실행 파일에도 device link 속성 켜기
set_target_properties(epi_test PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(epi_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
)

# ---- (선택) pybind 모듈 ----
option(BUILD_PYBIND "Build pybind11 module" OFF)
if(BUILD_PYBIND)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG QUIET)
  if(pybind11_FOUND)
    pybind11_add_module(_ops_epilogue
      pybind/epilogue_bind.cpp
    )
    target_link_libraries(_ops_epilogue PRIVATE epi CUDA::cudart)
    target_include_directories(_ops_epilogue PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(_ops_epilogue PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
    set_target_properties(_ops_epilogue PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
  else()
    message(WARNING "pybind11 not found; skipping Python module")
  endif()
endif()
