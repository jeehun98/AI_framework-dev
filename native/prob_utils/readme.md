# prob_utils CUDA Build Notes
*(Windows + NVCC 12.6 + Ninja + MSVC, project: `AI_framework-dev`)*

## TL;DR
- **에러 증상:** `calling a __device__ function("__int_as_float") from a __host__ function(...) is not allowed`  
- **근본 원인:** host 함수(예: `logsumexp_cuda`, `log_evidence_cuda`) 안에서 **device 전용 intrinsic** `__int_as_float`를 호출함.  
- **핵심 해결책:** host/device 모두에서 안전한 음의 무한대(-∞)를 쓰도록 정리  
  - `-CUDART_INF_F`(device/host에서 사용 가능, 헤더 필요) 또는  
  - 공용 헬퍼 `neg_inf()`로 **컴파일 경로 분기**

---

## 증상 (Symptoms)
빌드 중 NVCC가 아래와 같은 에러를 출력:
```
error: calling a __device__ function("__int_as_float") from a __host__ function("...") is not allowed
```
대표적으로 `bayes_cuda.cu` / `log_ops_cuda.cu`의 host 함수에서 발생.

원인 줄 예시(문제 코드):
```cpp
if (!x || !out || n==0) { if(out) *out = -__int_as_float(0x7f800000U); return; }
```

---

## 근본 원인 (Root Cause)
- `__int_as_float`는 NVCC가 제공하는 **device 전용 intrinsic** 이므로, **host** 함수에서 호출하면 컴파일 에러.
- 음의 무한대(-∞)를 얻기 위해 위와 같이 비표준적인 방법을 사용했던 것이 트리거.
- 추가로, 동일 레포가 **두 경로**(`C:\Users\owner\...` vs `C:\Users\as042\...`)에 존재해 빌드 대상 파일과 편집 파일이 달라진 것도 혼선을 키웠음.

---

## 해결 (Fix)
### 1) 공용 음의 무한대 헬퍼 도입
```cpp
#include <limits>
#include <cuda_runtime.h>
#include <math_constants.h>

__host__ __device__ inline float neg_inf() {
#if defined(__CUDA_ARCH__)
    return -CUDART_INF_F;                           // device 경로
#else
    return -std::numeric_limits<float>::infinity(); // host 경로
#endif
}
```
- **host** 경로: 표준 C++의 `-std::numeric_limits<float>::infinity()` 사용
- **device** 경로: `-CUDART_INF_F` 사용 (헤더: `<math_constants.h>` 필요)

**치환 규칙:**  
- `-__int_as_float(0x7f800000U)` → `neg_inf()` (또는 `-CUDART_INF_F`)

### 2) `log_ops_cuda.cu` 수정 요점
- `logsumexp_cuda`의 입력 검증:
```cpp
if (!x || !out || n == 0) { if (out) *out = neg_inf(); return; }
```
- 커널 `lse_reduce_kernel`에서 **all -∞ guard** 추가:
```cpp
if (maxv == -CUDART_INF_F) { if (tid == 0) *out = -CUDART_INF_F; return; }
```
- 선택: `__logf` vs `logf`는 `PROB_USE_FAST_MATH` 매크로로 분기

### 3) `bayes_cuda.cu` 수정 요점
- `log_evidence_cuda`의 **조기 반환**을 모두 `neg_inf()`로 통일
- `add_arrays_kernel` → **grid-stride loop**로 확장성 확보
- `posterior_kernel`에서 `logZ`가 `-inf`일 때 **NaN 전파 방지**:
  - `post_prob[i] = 0.f;` / `post_log[i] = -CUDART_INF_F`
- 헤더 포함 순서:
  1. `<cuda_runtime.h>`
  2. `<math_constants.h>` (for `CUDART_INF_F`)
  3. `<limits>` 등 표준 헤더
  4. 프로젝트 헤더

### 4) 잔여 사용처 일괄 제거
작업 루트에서 검색:
```powershell
Get-ChildItem -Recurse -Include *.cu,*.cuh,*.cpp,*.hpp | Select-String "__int_as_float"
```
발견 시 `neg_inf()` 또는 `-CUDART_INF_F`로 교체.

---

## 디버깅 보완
- 커널 런치 직후 에러 체크(디버그 빌드 한정):
```cpp
#ifndef NDEBUG
cudaError_t st = cudaGetLastError();
if (st != cudaSuccess) {
    // fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(st));
}
#endif
```
- `logsumexp`의 수치 안정성: all -∞ 입력 처리, `max-shift` 사용 완료

---

## 빌드/클린 절차
### 1) CMake/Ninja
```powershell
# 빌드 디렉터리에서
ninja -t clean
ninja
# 또는
cmake --build . --clean-first --config Release
```

### 2) CMake 초기화부터 (권장 재설정)
```powershell
cd C:\Users\as042\Desktop\AI_framework-dev\native\prob_utils\build
# 빌드 폴더를 비우거나 새로 만들기
Remove-Item * -Recurse -Force  # 주의: 실제 삭제
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_STANDARD=17
ninja
```

### 3) 경로 혼선 점검
동일 레포가 여러 사용자 폴더에 있을 경우, 실제로 **빌드되는 파일 경로**를 로그로 확인 후 편집 대상과 일치시킬 것.

---

## 자주 하는 실수 (Pitfalls)
- host 함수에서 device 전용 intrinsic 사용 (`__int_as_float`, `__float_as_int` 등)
- `<math_constants.h>` 미포함 상태에서 `CUDART_INF_F` 사용 → undeclared 에러
- VSCode에서 파일 **저장** 누락 후 빌드
- 여러 복제본 레포를 번갈아 수정
- `-rdc=true` 사용 시 링크/아카이브 단계 누락 (이번 건과 직접 연관은 적음)

---

## 확장 아이디어
- `logsumexp` 멀티블록/warp-shuffle 2패스 구현으로 큰 `n` 가속
- `double` 템플릿 버전 지원
- `posterior`와 `evidence`를 하나의 커널/스트림에서 pipeline 처리

---

## 최종 상태 체크
- 결과 샘플:
  - Evidence `P(B)` = `0.5`
  - Posterior `P(A|B)` = `0.84`
  - 검증: `P(A|B)P(B) = 0.42` == `P(B|A)P(A) = 0.42`
- Bayes 정리 일치 → 구현 정상 동작 확인
