# modelsel_core 빌드 가이드

## 📦 개요
`modelsel_core`는 **모델 선택 / 교차검증** 모듈을 독립적으로 구현한 C++17 라이브러리입니다.  
`discrete_core`와 분리되어 독립적으로 빌드 및 실행 가능합니다.

본 문서에서는 **정적 라이브러리(`.lib`) 빌드** 과정을 정리합니다. (DLL도 가능하나, 심볼 export 관리가 필요합니다.)

---

## 🛠️ 준비 환경
- Windows 10/11  
- Visual Studio 2022 (MSVC 19.42+)  
- CMake ≥ 3.22  
- Ninja (권장)  
- CUDA Toolkit ≥ 12.6 (옵션, CUDA evaluator 사용 시)  

---

## ⚙️ CMake 옵션 요약

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `BUILD_SHARED_LIBS` | **OFF** | ON이면 DLL, OFF면 정적 라이브러리 |
| `MODELSEL_WITH_CUDA` | OFF | CUDA evaluator 빌드 여부 |
| `BUILD_EXAMPLES` | ON | 예제(`modelsel_example.exe`) 빌드 여부 |

---

## 🏗️ 빌드 절차

### 1. CPU 전용 정적 빌드
```powershell
cd C:\Users\owner\Desktop\AI_framework-dev\native\modelsel_core

# 이전 빌드 캐시 제거
rd /s /q build_cpu 2>$null

# CMake 구성
cmake -S . -B build_cpu -G "Ninja" `
  -DCMAKE_BUILD_TYPE=Release `
  -DBUILD_EXAMPLES=ON `
  -DBUILD_SHARED_LIBS=OFF   # 정적 라이브러리 빌드

# 빌드
cmake --build build_cpu -v

# 실행
.\build_cpu\modelsel_example.exe
```

---

### 2. CUDA evaluator 포함 (선택)
```powershell
cd C:\Users\owner\Desktop\AI_framework-dev\native\modelsel_core

rd /s /q build_cuda 2>$null

cmake -S . -B build_cuda -G "Ninja" `
  -DCMAKE_BUILD_TYPE=Release `
  -DBUILD_EXAMPLES=ON `
  -DBUILD_SHARED_LIBS=OFF `
  -DMODELSEL_WITH_CUDA=ON `
  -DCMAKE_CUDA_ARCHITECTURES=86   # GPU 아키텍처 (예: RTX30 = 86)

cmake --build build_cuda -v
.\build_cuda\modelsel_example.exe
```

> ⚠️ `evaluator_cuda.cu`는 현재 placeholder 상태입니다. 실제 CUDA 평가를 쓰려면 discrete_core와 연동 구현 필요.

---

## 📂 빌드 결과

- **정적 라이브러리**:  
  - `build_cpu\modelsel_core.lib`  
  - (CUDA 빌드 시 `build_cuda\modelsel_core.lib`)  
- **예제 실행 파일**:  
  - `build_cpu\modelsel_example.exe`  
  - `build_cuda\modelsel_example.exe`  

---

## ✅ 실행 예시

```
==== Bernoulli 5-Fold CV Example ====
N=5000, p_true=0.3, estimator=MAP[Beta(2.000000,2.000000)], backend=cpu
[CV] k=5
 LogLoss: 0.611605 ± 0.0121579
 Accuracy: 0.6994 ± 0.0147919
```

- LogLoss ≈ 이론적 엔트로피(≈0.6109 nats)에 근접  
- Accuracy ≈ 기대 정확도(0.7)에 부합  

---

## 📌 요약
- **정적 빌드**는 가장 간단하고 DLL export 문제 없음.  
- CUDA는 옵션이며 현재는 CPU 평가가 기본.  
- 예제(`modelsel_example.exe`)를 통해 구현 검증 가능.  
