cmake_minimum_required(VERSION 3.22)
project(prob_utils LANGUAGES CXX)

option(BUILD_SHARED_LIBS "Build shared library" ON)
option(PROB_WITH_CUDA "Build CUDA backend" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (MSVC)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
    $<$<COMPILE_LANGUAGE:CXX>:/W4>
  )
endif()

add_library(prob_utils
  src/log_ops_cpu.cpp
  src/bayes_cpu.cpp
)
target_include_directories(prob_utils
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_definitions(prob_utils PRIVATE PROB_EXPORTS)

if (PROB_WITH_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  target_sources(prob_utils PRIVATE
    src/log_ops_cuda.cu
    src/bayes_cuda.cu
  )
  set_target_properties(prob_utils PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
  )
  target_include_directories(prob_utils PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  if (MSVC)
    target_compile_options(prob_utils PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
    )
  endif()
  target_compile_definitions(prob_utils PUBLIC PROB_WITH_CUDA)
  target_link_libraries(prob_utils PRIVATE CUDA::cudart)
endif()

add_executable(prob_example examples/main.cpp)
target_link_libraries(prob_example PRIVATE prob_utils)

# install (optional)
include(GNUInstallDirs)
install(TARGETS prob_utils
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
