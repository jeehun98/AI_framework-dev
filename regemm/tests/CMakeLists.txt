add_executable(test_basic tests/test_basic.cpp)
target_link_libraries(test_basic PRIVATE regemm cuda)
