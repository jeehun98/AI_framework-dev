🎨 NVTX Profiling & Visualization Guide

(Graph Executor v2 / CUDA NVTX Integration)

1️⃣ 목적

graph_executor_v2에서 각 연산(gemm, epilogue, bwd, 등)의 실행 구간을 시각화하기 위해
CUDA Graph Capture-safe한 NVTX_RANGE()/NVTX_MARK() 를 삽입하였다.

이 문서는 NVTX 기반 프로파일링을 Nsight Systems로 확인하는 전체 절차를 정리한 것이다.

2️⃣ NVTX 삽입 구조
#include "backends/cuda/ops/_common/shim/nvtx.hpp"

// 예시: forward launcher 내부
void launch_forward(...) {
    NVTX_RANGE("gemm.fwd", NVTX_COLOR::Orange);
    // kernel launch ...
}


nvtx.hpp 내부에서는 NVTX API 호출을 캡슐화:

#ifdef USE_NVTX
#include <nvToolsExt.h>
#define NVTX_RANGE(name, color) ::ai::nvtx::Range _nvtx_range_##__LINE__{name, static_cast<uint32_t>(color)};
#define NVTX_MARK(name) nvtxMarkA(name)
#endif


각 색상(NVTX_COLOR::Orange, Red, Cyan …)은 타임라인에서 시각적으로 구분된다.

3️⃣ 빌드 설정 (CMake)

CMakeLists.txt 내에서 다음 옵션이 활성화되어 있어야 한다:

option(USE_NVTX "Enable NVTX ranges" ON)

if (USE_NVTX)
  add_definitions(-DUSE_NVTX)
  if (WIN32)
    set(_NVTX_CANDIDATES nvToolsExt64_1 nvToolsExt)
  else()
    set(_NVTX_CANDIDATES nvToolsExt)
  endif()

  find_library(NVTOOLSEXT_LIB NAMES ${_NVTX_CANDIDATES}
               HINTS "$ENV{CUDA_PATH}/lib/x64" "$ENV{CUDA_PATH}/lib64")
  target_link_libraries(ai_backend_cuda PRIVATE ${NVTOOLSEXT_LIB})
endif()


빌드 명령 예시:

cmake -S . -B build -G "Ninja" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGE2_WITH_REGEMM=ON ^
  -DUSE_NVTX=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=86 ^
  -DPython3_EXECUTABLE="C:\Python312\python.exe" ^
  -Dpybind11_DIR="C:\Python312\Lib\site-packages\pybind11\share\cmake\pybind11"

cmake --build build --target _ops_gemm -j


빌드 완료 후 python/graph_executor_v2/ops/_ops_gemm.pyd 생성됨.

4️⃣ Nsight Systems 로 프로파일 수집

관리자 권한 CMD 또는 PowerShell에서:

"C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.5.1\target-windows-x64\nsys.exe" ^
  profile -o "C:\Users\owner\Desktop\AI_framework-dev\nsys_gemm" ^
  --trace=cuda,nvtx,cublas ^
  python "C:\Users\owner\Desktop\AI_framework-dev\graph_executor_v2\python\test\ops\gemm_nvtx_smoke.py"


성공 시 출력:

Generated:
    C:\Users\owner\Desktop\AI_framework-dev\nsys_gemm.nsys-rep

5️⃣ 리포트 열기 (GUI)

CLI와 같은 버전의 Nsight Systems GUI로 실행한다.

"C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.5.1\host-windows-x64\nsys-ui.exe" ^
  "C:\Users\owner\Desktop\AI_framework-dev\nsys_gemm.nsys-rep"


⚠️ 주의: .nsys-rep 파일을 더블클릭하면 구버전 Nsight Systems가 열릴 수 있음.
반드시 위 경로의 nsys-ui.exe 로 직접 실행.

6️⃣ NVTX 트랙 확인 방법

Nsight Systems GUI에서:

Trace → Show CUDA API Calls 활성화

Trace → Show NVTX Ranges 활성화

CPU 트랙(보통 Python 또는 ai_core 스레드) 하단에 NVTX 색상 블록 표시됨

예:

🟧 gemm.fwd

🔴 gemm.bwd

🟦 fallback.main_gemm

🟩 epilogue

7️⃣ 시각화 분석 포인트
분석 항목	설명
NVTX Range Layer	코드상 NVTX_RANGE()로 구간 태깅된 CPU–GPU 실행 범위
CUDA Kernels Track	실제 커널(gemm_bias_act_f32_tiled_kernel_ex, epilogue_fwd_kernel) 실행 시간
cuBLAS API Track	cublasSgemm_v2 등 BLAS 호출 구간
CPU Launch Overhead	NVTX 구간 사이의 빈 틈 (CPU scheduling delay)
Overlap 여부	CUDA Stream 병렬성 확인 (GEMM ↔ Epilogue 겹침)
8️⃣ 텍스트/HTML 리포트로 빠르게 보기

CLI 환경에서 요약 통계 출력:

nsys stats --report nvtxsum,gpukernsum,cublassum "nsys_gemm.nsys-rep" > nsys_gemm_stats.txt


HTML 내보내기:

nsys export --type html --output "nsys_gemm" "nsys_gemm.nsys-rep"


→ 생성된 nsys_gemm.html 파일을 브라우저에서 열면,
커널 타임라인 및 NVTX 범위를 시각적으로 확인 가능.

9️⃣ 예시 해석 (Forward pass)
커널 이름	비중	설명
gemm_bias_act_f32_tiled_kernel_ex	92%	메인 GEMM + Bias + Activation fused kernel
distribution_elementwise_grid_stride_kernel	6%	torch.randn 초기화 커널
epilogue_fwd_kernel	1%	후처리 활성화 커널 (별도 launch 시)
✅ 결론

NVTX 연동이 정상적으로 작동하면 CPU–GPU 경계의 실행 타임라인을 정확히 시각화 가능

gemm.fwd, gemm.bwd, epilogue 등의 구간별 성능 분리 및 병목 분석 용이

Nsight Systems 2024.5.1 GUI(nsys-ui.exe)에서 같은 버전으로 열어야 오류 없이 표시됨