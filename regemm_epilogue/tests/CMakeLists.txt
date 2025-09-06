add_executable(test_basic test_basic.cpp)
target_link_libraries(test_basic PRIVATE regemm)
# (선택) ctest 등록
add_test(NAME basic COMMAND test_basic)
