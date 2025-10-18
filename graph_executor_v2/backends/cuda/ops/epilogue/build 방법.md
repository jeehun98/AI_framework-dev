cd C:\Users\owner\Desktop\AI_framework-dev\graph_executor_v2\backends\cuda\ops\epilogue

cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86

cmake --build build -j

/build/epi_test  
# 출력: OK. errors=0
