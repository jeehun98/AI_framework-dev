cmake_minimum_required(VERSION 3.26)
project(graph_executor_v2 LANGUAGES CXX CUDA)

# ------------------------------ Python / pybind11 ------------------------------
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG QUIET)
if (NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.12.0
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# ------------------------------ CUDA / cuBLAS(Lt) ------------------------------
find_package(CUDAToolkit REQUIRED)

# 목적 GPU 아키텍처 (필요 시 -DCMAKE_CUDA_ARCHITECTURES="86;89" 로 오버라이드)
if (NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# (선택) NVTX 사용 옵션
option(GE2_WITH_NVTX "Link NVTX for profiling ranges" OFF)

# ------------------------------ regemm_epilogue 통합 ---------------------------
# 레포 구조:
#   AI_framework-dev/
#     ┣ regemm_epilogue/
#     ┗ graph_executor_v2/
option(GE2_WITH_REGEMM "Integrate regemm_epilogue" ON)

if (GE2_WITH_REGEMM)
  if (EXISTS "${CMAKE_CURRENT_LIST_DIR}/../regemm_epilogue/CMakeLists.txt")
    add_subdirectory(../regemm_epilogue ${CMAKE_BINARY_DIR}/regemm_epilogue_build)
    set(REGEMM_TARGET regemm_epilogue)
    set(REGEMM_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../regemm_epilogue/include)
  else()
    message(FATAL_ERROR "regemm_epilogue not found next to graph_executor_v2")
  endif()
endif()

# ------------------------------ 소스 목록 ------------------------------
set(SRC_CORE
  src/launch_table.cpp
  src/my_kernels.cu
)
set(SRC_PY
  src/bindings_min_api.cpp
)

# ------------------------------ 코어 라이브러리 ------------------------------
add_library(ge_v2_core STATIC ${SRC_CORE})
target_compile_features(ge_v2_core PRIVATE cxx_std_17)
set_target_properties(ge_v2_core PROPERTIES
  CUDA_STANDARD                17
  CUDA_SEPARABLE_COMPILATION   ON
  POSITION_INDEPENDENT_CODE    ON
)

target_include_directories(ge_v2_core PRIVATE
  ${CUDAToolkit_INCLUDE_DIRS}
  ${CMAKE_CURRENT_LIST_DIR}/include
  $<$<BOOL:${GE2_WITH_REGEMM}>:${REGEMM_INCLUDE_DIR}>
)

target_link_libraries(ge_v2_core PRIVATE
  CUDA::cudart
  CUDA::cublas
  CUDA::cublasLt
  $<$<BOOL:${GE2_WITH_NVTX}>:CUDA::nvToolsExt>
  $<$<BOOL:${GE2_WITH_REGEMM}>:${REGEMM_TARGET}>
)

# ------------------------------ 파이썬 모듈 ------------------------------
pybind11_add_module(graph_executor_v2 MODULE ${SRC_PY})
target_compile_features(graph_executor_v2 PRIVATE cxx_std_17)
set_target_properties(graph_executor_v2 PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME "graph_executor_v2"
)

# MSVC/Unix 공용 컴파일 옵션
if (MSVC)
  target_compile_options(ge_v2_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/bigobj;/W3;/EHsc;/O2;/MD>
  )
  target_compile_options(graph_executor_v2 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/bigobj;/W3;/EHsc;/O2;/MD>
  )
else()
  target_compile_options(ge_v2_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-O2>
  )
  target_compile_options(graph_executor_v2 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-O2>
  )
endif()

# 모듈은 코어에 링크
target_link_libraries(graph_executor_v2 PRIVATE ge_v2_core)

# 파이썬 모듈이 헤더에 접근 필요하면 include 추가
target_include_directories(graph_executor_v2 PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/include
)

# Windows 에서 .pyd 접미사 확실히
if (WIN32)
  set_target_properties(graph_executor_v2 PROPERTIES SUFFIX ".pyd")
endif()

# ------------------------------ 설치 규칙(선택) ------------------------------
install(TARGETS graph_executor_v2
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .
        ARCHIVE DESTINATION .)

# ✅ EX API도 같이 설치
install(FILES
  include/ge_v2_api.h
  include/ge_v2_api_ex.h
  DESTINATION include
)

# ------------------------------ 테스트/편의(선택) ------------------------------
enable_testing()
add_test(NAME import_test COMMAND ${Python_EXECUTABLE} -c "import graph_executor_v2; print('ok')")

# 빌드 후 test/로 결과 모듈 복사(개발 편의)
set(_GE2_OUT $<TARGET_FILE:graph_executor_v2>)
add_custom_command(TARGET graph_executor_v2 POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${_GE2_OUT}
          ${CMAKE_CURRENT_LIST_DIR}/test/$<TARGET_FILE_NAME:graph_executor_v2>
  COMMENT "Copying graph_executor_v2 to test/ for quick import"
)

# ------------------------------ 빌드 메모 ------------------------------
# 1) Windows(Ninja):
#    cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DGE2_WITH_REGEMM=ON -DCMAKE_CUDA_ARCHITECTURES=86
#    ninja
# 2) Windows(MSVC):
#    cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DGE2_WITH_REGEMM=ON
#    cmake --build build --config Release
# 3) Linux:
#    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGE2_WITH_REGEMM=ON -DCMAKE_CUDA_ARCHITECTURES="86;89"
#    cmake --build build -j
#
# 런타임:
#  - Windows: cudart64_*.dll, cublas*.dll, cublasLt*.dll 이 PATH에 있어야 함.
#  - Linux: 통상 /usr/lib/x86_64-linux-gnu 등. 미설치면 LD_LIBRARY_PATH 준비.
