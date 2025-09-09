cmake_minimum_required(VERSION 3.22)
project(modelsel_core LANGUAGES CXX)

option(BUILD_SHARED_LIBS "Build shared library" ON)
option(MODELSEL_WITH_CUDA "Build CUDA evaluator (optional)" OFF)
option(BUILD_EXAMPLES "Build examples" ON)

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

# 출력 경로 고정
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(MODELSEL_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(MODELSEL_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(EXAMPLES_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/examples)

set(MODELSEL_SOURCES
  ${MODELSEL_SRC_DIR}/split.cpp
  ${MODELSEL_SRC_DIR}/kfold.cpp
  ${MODELSEL_SRC_DIR}/metrics.cpp
  ${MODELSEL_SRC_DIR}/aicbic.cpp
  ${MODELSEL_SRC_DIR}/estimator.cpp
  ${MODELSEL_SRC_DIR}/evaluator_cpu.cpp
  ${MODELSEL_SRC_DIR}/report.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/modelsel_c.cpp
)

if (MODELSEL_WITH_CUDA)
  enable_language(CUDA)
  if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
  endif()
  list(APPEND MODELSEL_SOURCES ${MODELSEL_SRC_DIR}/evaluator_cuda.cu)
  set_source_files_properties(${MODELSEL_SRC_DIR}/evaluator_cuda.cu PROPERTIES LANGUAGE CUDA)
  set_target_properties(modelsel_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
  )
endif()

add_library(modelsel_core ${MODELSEL_SOURCES})
target_include_directories(modelsel_core PUBLIC ${MODELSEL_INC_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

if (BUILD_EXAMPLES)
  add_executable(modelsel_example ${EXAMPLES_DIR}/main.cpp)
  target_link_libraries(modelsel_example PRIVATE modelsel_core)
  target_include_directories(modelsel_example PRIVATE ${MODELSEL_INC_DIR})
endif()
