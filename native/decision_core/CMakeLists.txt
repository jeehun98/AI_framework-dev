cmake_minimum_required(VERSION 3.22)
project(decision_core LANGUAGES CXX)

# C++17 / 정적 라이브러리 기본
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build shared library" OFF)
option(BUILD_EXAMPLES "Build examples" ON)

add_library(decision_core
  src/decision.cpp
  src/loss.cpp
  src/reject.cpp
  src/regression.cpp
)

target_include_directories(decision_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if (BUILD_EXAMPLES)
  add_executable(decision_example examples/main.cpp)
  target_link_libraries(decision_example PRIVATE decision_core)
  target_include_directories(decision_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

# 설치(선택)
include(GNUInstallDirs)
install(TARGETS decision_core
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
