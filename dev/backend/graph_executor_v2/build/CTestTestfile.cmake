# CMake generated Testfile for 
# Source directory: C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor_v2
# Build directory: C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor_v2/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[import_test]=] "C:/Users/owner/AppData/Local/Programs/Python/Python312/python.exe" "-c" "import graph_executor_v2; print('ok')")
set_tests_properties([=[import_test]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor_v2/CMakeLists.txt;136;add_test;C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor_v2/CMakeLists.txt;0;")
subdirs("regemm_epilogue_build")
