cmake_minimum_required(VERSION 3.26)
project(graph_executor_v2 LANGUAGES CXX CUDA)

# Python/pybind11/CUDA
find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# 아키텍처는 CMake 표준 변수로 지정 (86 = Ampere)
# 여러 개면 세미콜론으로 나열: e.g. 75;86;89
set(CMAKE_CUDA_ARCHITECTURES 86)

set(SRC
  src/bindings_min_api.cpp
  src/launch_table.cpp
  src/my_kernels.cu
)

# 파이썬 확장 모듈 생성
pybind11_add_module(graph_executor_v2 MODULE ${SRC})

# 표준 설정
target_compile_features(graph_executor_v2 PRIVATE cxx_std_17)
set_target_properties(graph_executor_v2 PROPERTIES
  CUDA_STANDARD 17
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME "graph_executor_v2"
)

# MSVC 전용 옵션은 CXX에만 적용 (CUDA에는 직접 넣지 않음)
if (MSVC)
  target_compile_options(graph_executor_v2 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/bigobj;/W3;/EHsc;/O2;/MD>
  )
else()
  target_compile_options(graph_executor_v2 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-O2>
  )
endif()

# CUDA 전용 호스트컴파일러 옵션은 -Xcompiler 로 전달
target_compile_options(graph_executor_v2 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc;-Xcompiler=/O2;-Xcompiler=/MD>
)

# 링크
target_link_libraries(graph_executor_v2 PRIVATE
  CUDA::cudart
  CUDA::cublas
  # CUDA::cublasLt  # 필요 시 활성화
  # Python::Module  # 보통 pybind11_add_module이 처리하지만, 환경에 따라 필요할 수 있음
)

# 인클루드 경로
target_include_directories(graph_executor_v2 PRIVATE
  ${CUDAToolkit_INCLUDE_DIRS}
  include
  # ${Python_INCLUDE_DIRS}  # 보통 불필요. 남겨도 무방.
)

# 설치: 모듈(.pyd/.so) + 공개 헤더(ABI 명세)
install(TARGETS graph_executor_v2
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .
        ARCHIVE DESTINATION .)

install(FILES include/ge_v2_api.h DESTINATION include)
