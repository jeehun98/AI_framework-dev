rmdir /s /q build
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DGE2_WITH_REGEMM=ON -DCMAKE_CUDA_ARCHITECTURES=86 ^
  -DPython_EXECUTABLE="C:\Users\owner\AppData\Local\Programs\Python\Python312\python.exe" ^
  -DPython_ROOT_DIR="C:\Users\owner\AppData\Local\Programs\Python\Python312" ^
  -Dpybind11_DIR="C:\Users\owner\AppData\Local\Programs\Python\Python312\Lib\site-packages\pybind11\share\cmake\pybind11"
cmake --build build -j

cmake -S . -B build -G "Ninja" ^
More?   -DCMAKE_BUILD_TYPE=Release ^
More?   -DGE2_WITH_REGEMM=ON ^
More?   -DCMAKE_CUDA_ARCHITECTURES=86 ^
More?   -DPYBIND11_FINDPYTHON=ON ^
More?   -DPython3_ROOT_DIR="C:/Users/owner/AppData/Local/Programs/Python/Python312" ^
More?   -DPython3_EXECUTABLE="C:/Users/owner/AppData/Local/Programs/Python/Python312/python.exe" ^
More?   -DPython3_FIND_STRATEGY=LOCATION ^
More?   -DPython3_FIND_VIRTUALENV=STANDARD


# graph_executor_v2 — 빌드 가이드 (Windows & Linux, 한글)

**목적**: `graph_executor_v2`를 **regemm_epilogue**와 연동하여 Python 확장 모듈로 빌드하는 방법을 한 번에 정리했습니다.  
(다음에 또 빌드할 때 이 문서만 보면 됩니다.)

---

## 0) 디렉터리 구성(예시)

아래와 같은 레이아웃을 권장합니다. 실제 위치가 다르면 CMake 옵션 `REGEMM_EPILOGUE_DIR`로 경로를 지정하세요.

```
AI_framework-dev/
 ┣ regemm_epilogue/                 # ← regemm 프로젝트
 ┗ dev/backend/graph_executor_v2/   # ← 이 프로젝트
```

---

## 1) 준비물

- **CUDA Toolkit** (nvcc, cudart) + 지원 GPU
- **Visual Studio 2022** C++ 도구(Windows) + **Windows SDK**
  - 구성 요소: *MSVC v143*, *Windows 10/11 SDK*, *C++ CMake tools for Windows*
- **Python 3.8+** (예시에서는 3.12 사용)
- **pybind11** (Python 패키지)
  ```powershell
  python -m pip install -U pybind11
  ```
- **CMake 3.26+**, **Ninja**(Ninja 제너레이터를 사용할 경우)

> 💡 Windows에서는 **“x64 Native Tools Command Prompt for VS 2022”** 혹은 **Developer PowerShell**에서 빌드하면 `rc.exe`/`mt.exe`를 자동으로 찾습니다.

---

## 2) 주요 CMake 옵션 요약

- `-DREGEMM_EPILOGUE_DIR="C:/.../regemm_epilogue"` : regemm 루트 경로(내부에 `CMakeLists.txt` 존재)
- `-DGE2_WITH_REGEMM=ON` : regemm 연동 사용 (기본 ON)
- `-DCMAKE_CUDA_ARCHITECTURES=86` : 타겟 SM 지정 (Ampere=86, Ada=89 등 `86;89` 형태도 가능)
- `-DPython_EXECUTABLE="C:/.../Python312/python.exe"` : 사용할 파이썬 명시
- `-Dpybind11_DIR="C:/.../site-packages/pybind11/share/cmake/pybind11"` : pybind11 CMake 패키지 경로

---

## 3) 빌드 — Windows (Ninja 제너레이터)

**VS 2022 개발자 명령 프롬프트**에서 실행 권장:

```bat
cd dev\backend\graph_executor_v2
rmdir /s /q build

set PY=C:\Users\owner\AppData\Local\Programs\Python\Python312\python.exe
set PYROOT=C:\Users\owner\AppData\Local\Programs\Python\Python312
set PYB11=C:\Users\owner\AppData\Local\Programs\Python\Python312\Lib\site-packages\pybind11\share\cmake\pybind11
set REGEMM=C:\Users\owner\Desktop\AI_framework-dev\regemm_epilogue

cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release ^
  -DGE2_WITH_REGEMM=ON -DREGEMM_EPILOGUE_DIR="%REGEMM%" -DCMAKE_CUDA_ARCHITECTURES=86 ^
  -DPython_EXECUTABLE="%PY%" -DPython_ROOT_DIR="%PYROOT%" -Dpybind11_DIR="%PYB11%"

cmake --build build -j
ctest --test-dir build -V
```

> ⚠️ `CMAKE_MT-NOTFOUND` 또는 `rc ... no such file or directory` 에러가 나면 **반드시** 개발자 명령 프롬프트에서 실행하세요.

---

## 4) 빌드 — Windows (Visual Studio 제너레이터)

```bat
cd dev\backend\graph_executor_v2
rmdir /s /q build

set PY=C:\Users\owner\AppData\Local\Programs\Python\Python312\python.exe
set PYROOT=C:\Users\owner\AppData\Local\Programs\Python\Python312
set PYB11=C:\Users\owner\AppData\Local\Programs\Python\Python312\Lib\site-packages\pybind11\share\cmake\pybind11
set REGEMM=C:\Users\owner\Desktop\AI_framework-dev\regemm_epilogue

cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
  -DGE2_WITH_REGEMM=ON -DREGEMM_EPILOGUE_DIR="%REGEMM%" -DCMAKE_CUDA_ARCHITECTURES=86 ^
  -DPython_EXECUTABLE="%PY%" -DPython_ROOT_DIR="%PYROOT%" -Dpybind11_DIR="%PYB11%"

cmake --build build --config Release
ctest --test-dir build -C Release -V
```

---

## 5) 빌드 — Linux

```bash
cd dev/backend/graph_executor_v2
rm -rf build

PY=$(which python3)
PYB11=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
REGEMM=../../regemm_epilogue

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   -DGE2_WITH_REGEMM=ON -DREGEMM_EPILOGUE_DIR="${REGEMM}" -DCMAKE_CUDA_ARCHITECTURES="86;89"   -DPython_EXECUTABLE="${PY}" -Dpybind11_DIR="${PYB11}"

cmake --build build -j
ctest --test-dir build -V
```

---

## 6) 실행(파이썬)

빌드가 완료되면 모듈 이름은 `graph_executor_v2` 입니다.

```powershell
python -c "import graph_executor_v2; print('import ok')"
```

> 실제 커널을 돌리려면 `gemm_bias_act_f32(A,B,D,bias, params)`에 **CUDA 디바이스 포인터**(정수 주소 또는 capsule)를 넘겨야 합니다.

---

## 7) 트러블슈팅

### A) `pybind11Config.cmake` 를 찾지 못함
```powershell
python -m pip install -U pybind11
python -c "import pybind11; print(pybind11.get_cmake_dir())"
cmake ... -Dpybind11_DIR="<<위에서 출력된 경로>>"
```
- CMake가 **같은 파이썬**을 쓰도록 `-DPython_EXECUTABLE=...`도 같이 넘겨주세요.

### B) `CMAKE_MT-NOTFOUND` / `rc ... no such file or directory` (Windows)
- **x64 Native Tools Command Prompt for VS 2022**에서 실행하거나, **Visual Studio 제너레이터** 사용.
- Windows SDK 설치 확인.

### C) `regemm_epilogue not found`
- 경로 확인 후 명시적으로 전달:
  ```
  -DREGEMM_EPILOGUE_DIR="C:/.../regemm_epilogue"
  ```

### D) 파이썬 버전 혼선(MSYS vs CPython 등)
- 다음 변수를 강제 지정:
  ```
  -DPython_EXECUTABLE="C:/.../Python312/python.exe"
  -DPython_ROOT_DIR="C:/.../Python312"
  -Dpybind11_DIR="C:/.../site-packages/pybind11/share/cmake/pybind11"
  ```
- 재설정 전 `build/` 폴더 삭제.

### E) CUDA 아키텍처 미스매치
- GPU에 맞게 `-DCMAKE_CUDA_ARCHITECTURES=86`(Ampere), `89`(Ada) 등으로 맞추세요.

---

## 8) 비고

- 이 프로젝트는 파이썬 확장 모듈 `graph_executor_v2`를 빌드하며, 내부에서 정적 라이브러리 `ge_v2_core`를 링크합니다.
- **FP32 GEMM 경로는 regemm_epilogue로 라우팅**됩니다.  
  FP16(cuBLASLt)은 현재 스텁이며, 필요 시 기존 Lt 경로를 연결하세요.
- Windows의 경우, import 시 `cudart64_*.dll`, `cublas*.dll`, `cublasLt*.dll`이 `PATH`에 있어야 합니다.

---

행복한 빌드 되세요! 🚀
