cmake_minimum_required(VERSION 3.22)
project(discrete_core LANGUAGES CXX)

option(BUILD_SHARED_LIBS "Build shared library" ON)
option(DISCRETE_WITH_CUDA "Build CUDA backend" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# MSVC: CXX에만 적용
if (MSVC)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
    $<$<COMPILE_LANGUAGE:CXX>:/W4>
  )
endif()

add_library(discrete_core
  src/bernoulli_cpu.cpp
  src/binomial_cpu.cpp
)

target_include_directories(discrete_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_definitions(discrete_core PRIVATE DISCRETE_EXPORTS)

if (DISCRETE_WITH_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
  endif()

  target_sources(discrete_core PRIVATE
    src/bernoulli_cuda.cu
    src/binomial_cuda.cu
  )

  set_target_properties(discrete_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
  )

  target_include_directories(discrete_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

  if (MSVC)
    target_compile_options(discrete_core PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
    )
  endif()

  target_compile_definitions(discrete_core PUBLIC DISCRETE_WITH_CUDA)
  target_link_libraries(discrete_core PRIVATE CUDA::cudart)
endif()

add_executable(discrete_example examples/main.cpp)
target_link_libraries(discrete_example PRIVATE discrete_core)

# install (optional)
include(GNUInstallDirs)
install(TARGETS discrete_core
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
