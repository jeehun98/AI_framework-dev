cmake_minimum_required(VERSION 3.22)
project(normal_core LANGUAGES CXX)

# --- Options ---------------------------------------------------------------
option(BUILD_SHARED_LIBS "Build shared library" ON)
option(NORMAL_WITH_CUDA "Build CUDA backend" OFF)

# --- Globals ---------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# MSVC: C++ 컴파일(=CXX)에만 인코딩/매크로/경고 옵션 적용
if (MSVC)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
    $<$<COMPILE_LANGUAGE:CXX>:/W4>
  )
endif()

# --- Core library (CPU) ----------------------------------------------------
add_library(normal_core
  src/normal_cpu.cpp
)

target_include_directories(normal_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Windows에서 내보내기 심볼 정의
target_compile_definitions(normal_core PRIVATE NORMAL_EXPORTS)

# --- Optional: CUDA backend ------------------------------------------------
if (NORMAL_WITH_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  # 기본 아키텍처(원하면 -DCMAKE_CUDA_ARCHITECTURES=86 등으로 덮어쓰기)
  if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
  endif()

  add_library(normal_cuda OBJECT src/normal_cuda.cu)
  set_target_properties(normal_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
  )

  # nvcc가 헤더를 찾도록 include 경로 부여
  target_include_directories(normal_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  # nvcc -> MSVC 옵션은 -Xcompiler로 각각 전달
  if (MSVC)
    target_compile_options(normal_cuda PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
    )
  endif()

  # DLL export/조건부 API 활성화를 위해 정의 추가
  target_compile_definitions(normal_cuda PRIVATE
    NORMAL_WITH_CUDA
    NORMAL_EXPORTS
  )
  target_compile_definitions(normal_core PUBLIC NORMAL_WITH_CUDA)

  # CUDA OBJECT 파일을 최종 타깃 소스로 병합 + device 심볼 해결
  target_sources(normal_core PRIVATE $<TARGET_OBJECTS:normal_cuda>)
  set_target_properties(normal_core PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  # 공식 CUDAToolkit 타깃 사용(런타임)
  target_link_libraries(normal_core PRIVATE CUDA::cudart)
endif()

# --- Example executable ----------------------------------------------------
add_executable(normal_example examples/main.cpp)
target_link_libraries(normal_example PRIVATE normal_core)

# --- Install rules (optional) ---------------------------------------------
include(GNUInstallDirs)
install(TARGETS normal_core
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
