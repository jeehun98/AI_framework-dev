# 📦 regemm_epilogue

GEMM(General Matrix Multiplication) 연산 이후 **epilogue 단계** (bias 추가, activation 적용 등)를 모듈화하여 고성능으로 구현한 라이브러리.

---

## 📂 프로젝트 구조

```plaintext
regemm_epilogue
 ┣ 📂benchmarks
 ┃ ┣ 📜benchmarks.md
 ┃ ┗ 📜bench_gemm.cu
 ┣ 📂docs
 ┃ ┣ 📜00_overview.md
 ┃ ┣ 📜10_design.md
 ┃ ┗ 📜20_perf_checklist.md
 ┣ 📂include
 ┃ ┗ 📂regemm
 ┃ ┃ ┣ 📜activations.h
 ┃ ┃ ┣ 📜api.h
 ┃ ┃ ┣ 📜bias.h
 ┃ ┃ ┣ 📜config.h
 ┃ ┃ ┗ 📜nvtx_shim.h
 ┣ 📂scripts
 ┃ ┣ 📜build.bat
 ┃ ┣ 📜build.sh
 ┃ ┗ 📜nsight_compute.sh
 ┣ 📂src
 ┃ ┣ 📜gemm_bias_act_smoke.cu
 ┃ ┣ 📜gemm_bias_act_tiled.cu
 ┃ ┗ 📜launcher.cu
 ┣ 📂tests
 ┃ ┣ 📜CMakeLists.txt
 ┃ ┗ 📜test_basic.cpp
 ┣ 📜.gitignore
 ┣ 📜build.md
 ┗ 📜CMakeLists.txt
```

---

## 📑 디렉터리/파일 설명

### 🔹 `benchmarks/`
- **`bench_gemm.cu`** : GEMM + epilogue 성능 벤치마크용 커널.
- **`benchmarks.md`** : 벤치마크 수행 방법 및 결과 기록.

### 🔹 `docs/`
- **`00_overview.md`** : 프로젝트 개요.
- **`10_design.md`** : 설계 원리 및 epilogue 모듈화 방식.
- **`20_perf_checklist.md`** : 성능 최적화 체크리스트 (레지스터/메모리/occupancy 등).

### 🔹 `include/regemm/`
라이브러리 공용 API 헤더.
- **`activations.h`** : ReLU, GELU 등 활성화 함수 정의.
- **`api.h`** : 외부 호출용 런처 및 API.
- **`bias.h`** : bias 연산 정의.
- **`config.h`** : 매크로 및 설정값 (스레드 블록, 타일 크기 등).
- **`nvtx_shim.h`** : NVTX 마커 래퍼 (프로파일링 지원).

### 🔹 `scripts/`
빌드 및 프로파일링 스크립트.
- **`build.bat`** : Windows 환경 빌드.
- **`build.sh`** : Linux 환경 빌드.
- **`nsight_compute.sh`** : Nsight Compute 실행 스크립트.

### 🔹 `src/`
CUDA 커널 구현부.
- **`gemm_bias_act_smoke.cu`** : 최소 단위 smoke test 커널.
- **`gemm_bias_act_tiled.cu`** : 타일링 기반 고성능 epilogue 커널.
- **`launcher.cu`** : 템플릿 런처, API와 커널 연결.

### 🔹 `tests/`
기능 및 correctness 테스트.
- **`test_basic.cpp`** : bias/activation 적용 결과 검증.
- **`CMakeLists.txt`** : 테스트 빌드 정의.

### 🔹 루트 파일
- **`.gitignore`** : 빌드 산출물 무시 규칙.
- **`build.md`** : 빌드 가이드 (환경 설정, CMake 사용법).
- **`CMakeLists.txt`** : 프로젝트 메인 빌드 스크립트.

---

## 🚀 사용 흐름
1. `include/regemm/api.h` 를 통해 GEMM + epilogue 호출.  
2. 내부적으로 `src/launcher.cu` → `gemm_bias_act_tiled.cu` 실행.  
3. 성능 검증은 `benchmarks/bench_gemm.cu` 사용.  
4. correctness 검증은 `tests/test_basic.cpp` 실행.  

---

## 📌 특징
- **Epilogue 모듈화** : bias, activation 등 후처리를 독립 모듈로 구현.  
- **고성능 타일링** : shared memory 및 레지스터 활용 최적화.  
- **크로스 플랫폼 빌드** : Windows/Linux 모두 지원.  
- **프로파일링 친화적** : NVTX 마커, Nsight Compute 스크립트 내장.  

---
