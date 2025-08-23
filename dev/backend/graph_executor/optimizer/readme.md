# optimizer — SGD / Momentum / Adam (CUDA)

본 모듈은 파라미터 업데이트 커널과 런처를 제공하며, 선택적으로 **Decoupled Weight Decay(AdamW/SGD‑WD)**, **Nesterov Momentum**, **Gradient Clipping**(값/글로벌 노름), **AMSGrad** 등을 지원합니다.

---

## 1) 파일 구조

* **`optimizer_types.cuh`**

  * `enum class OptimizerType { SGD, MOMENTUM, ADAM };`
  * 문자열/정수 변환 유틸 및 디버그용 출력 연산자.
* **`optimizer_config.cuh`**

  * 기능 토글 매크로 정의 (빌드 전역에서 제어 가능)

    * `WEIGHT_DECAY_ENABLE` (0/1)
    * `NESTEROV_ENABLE` (0/1)
    * `AMSGRAD_ENABLE` (0/1)
    * `GRAD_CLIP_ENABLE` (0/1) + `GRAD_CLIP_THRESH`
    * `GLOBAL_NORM_CLIP_ENABLE` (0/1)
    * `DEBUG_KERNEL` (0/1)
* **`optimizer_kernels.cu` / `optimizer_kernels.cuh`**

  * 커널 구현(SGD/Momentum/Adam) 및 **호스트 런처** `optimizer_update_cuda(...)`.
  * (선택) 글로벌 노름 클리핑용 리덕션/스케일 커널 포함.

> **빌드 팁:** 기능 토글은 `optimizer_config.cuh`에서 기본값을 두되, 필요 시 NVCC 컴파일 옵션으로 `-DWEIGHT_DECAY_ENABLE=1` 처럼 재정의하세요.

---

## 2) 퍼블릭 API (호스트 런처)

```cpp
void optimizer_update_cuda(
    float* param,
    const float* grad,
    float* velocity,   // MOMENTUM에서만 사용 (nullable)
    float* m,          // ADAM에서만 사용 (nullable)
    float* v,          // ADAM에서만 사용 (nullable)
#if AMSGRAD_ENABLE
    float* vhat_max,   // AMSGrad 최대치 (nullable)
#endif
    float lr, float beta1, float beta2, float eps,
#if WEIGHT_DECAY_ENABLE
    float weight_decay,
#endif
    int size,
    OptimizerType opt_type,
    int timestep,
    cudaStream_t stream);
```

* `param, grad` 크기: **`size`(= rows\*cols)**
* **SGD**: `param, grad`만 필요.
* **Momentum**: `velocity` 필요 (`beta1`를 모멘텀 계수로 사용).
* **Adam/AdamW**: `m, v` 필요 (`beta1, beta2, eps, timestep` 사용).
* **Decoupled WD**: `WEIGHT_DECAY_ENABLE=1`일 때 `weight_decay` 적용.
* **AMSGrad**: `AMSGRAD_ENABLE=1`이면 `vhat_max` 사용.

---

## 3) 내부 커널 요약

```cpp
__global__ void sgd_kernel(float* param, const float* grad,
#if WEIGHT_DECAY_ENABLE
                           float weight_decay,
#endif
                           float lr, int n);

__global__ void momentum_kernel(float* param, const float* grad, float* velocity,
#if WEIGHT_DECAY_ENABLE
                                float weight_decay,
#endif
                                float lr, float beta, int n);

__global__ void adam_kernel(float* param, const float* grad, float* m, float* v,
#if AMSGRAD_ENABLE
                            float* vhat_max,
#endif
#if WEIGHT_DECAY_ENABLE
                            float weight_decay,
#endif
                            float lr, float b1, float b2, float eps,
                            int t, int n);
```

* **값 클리핑**: `GRAD_CLIP_ENABLE=1`이면 `[-T, +T]`로 clamp (`T=GRAD_CLIP_THRESH`).
* **글로벌 노름 클리핑**: `GLOBAL_NORM_CLIP_ENABLE=1`이면 `∥g∥₂` 계산 후 스케일 적용(2‑pass). 성능상 워크스페이스/스트림 처리에 유의.
* **Weight Decay**: 모두 **Decoupled** 방식(AdamW/SGD‑WD). 즉, `p -= lr*wd*p`가 grad 적용과 분리되어 추가됩니다.

---

## 4) 수치/안정성 규칙

* 모든 커널에서 NaN/Inf grad는 무시(`continue`).
* Adam bias‑correction: `m̂ = m/(1-β₁ᵗ)`, `v̂ = v/(1-β₂ᵗ)`.
* 분모 보호: `sqrt(max(v̂, 1e-12)) + eps`.
* `value_clip()`는 클리핑 + NaN 가드.

---

## 5) 스트림/런치 규약 (**중요**)

* 런처 인자로 받은 `stream`을 **모든 커널 런치에 전달**해야 합니다.

  * 예: `sgd_kernel<<<blocks, threads, 0, stream>>>(...)`.
* 글로벌 노름 클리핑 경로의 `grad_sqsum_kernel`, `scale_grad_kernel`도 반드시 같은 `stream` 사용.

---

## 6) 메모리/버퍼 계약

* `param/grad/velocity/m/v/vhat_max`는 모두 **float32, C‑contiguous**.
* **크기는 동일:** `size = rows*cols`.
* Python 측에서 파라미터별 버퍼를 보관하며, 런처에 포인터를 전달.
* **글로벌 노름 클리핑**의 임시 버퍼(예: `grad_scaled`, `d_partial`):

  * 현재 샘플 구현은 함수 내부 `static`/`cudaMalloc`을 사용 → **멀티‑스트림/멀티‑파라미터 동시 업데이트 시 레이스** 위험.
  * 권장: (1) 외부에서 워크스페이스 전달 인터페이스 추가, (2) 파라미터별 scratch, (3) 스트림별 allocator 사용.

---

## 7) 실행기(executor) 연동 규칙

* `train_step_entry`에서 학습 대상 파라미터 ID를 수집 후, 동일 ID의

  * `tensors[name]` → `param`
  * `gradients[name]` → `grad`
  * `velocity_ptrs[name]` / `m_ptrs[name]` / `v_ptrs[name]` / `vhat_max_ptrs[name]` → 옵티마 상태
  * `shapes[name]` → `size = rows*cols`
* 옵티마 옵션은 Python에서 선택:

  * `opt_type ∈ {SGD, MOMENTUM, ADAM}`
  * `lr, beta1, beta2, eps, timestep`
  * (옵션) `weight_decay`

---

## 8) 성능/튜닝 메모

* 스레드/블록: 기본 `threads=256`, `blocks=ceil(n/256)`.
* 벡터화(`float2/float4`)는 메모리 정렬 시 이득.
* 여러 파라미터를 한 커널로 **fuse**(multi‑tensor apply)하면 런치 오버헤드를 절감할 수 있음(향후 작업).
* 글로벌 노름 클리핑은 2‑pass + 호스트 복사로 비용 큼 → 큰 모델에서는 비활성 또는 전용 리덕션으로 교체 권장.

---

## 9) 흔한 실수 / 체크리스트

* [ ] `optimizer_config.cuh`의 매크로 오탈자: `#define GLOBAL_NORM_CLIP_ENABLE 0` **(쉼표 금지)**
* [ ] 런처에서 `stream` 인자를 전달하지 않음 → **기본 스트림**으로 실행됨
* [ ] 파라미터별 버퍼 크기 불일치(`size`) → out‑of‑bounds 위험
* [ ] AMSGrad/Vhat 미제공인데 `AMSGRAD_ENABLE=1`
* [ ] `velocity/m/v` 포인터 누락 → MOMENTUM/ADAM에서 업데이트 skip
* [ ] Python이 전달하는 학습률/β/eps/timestep 값의 범위 확인(특히 `timestep>=1`)

---

## 10) 예시: AdamW 업데이트 호출 (의사코드)

```cpp
// param, grad, m, v, vhat_max: device pointers (size = rows*cols)
optimizer_update_cuda(
    param, grad,
    /*velocity=*/nullptr,
    m, v,
#if AMSGRAD_ENABLE
    vhat_max,
#endif
    /*lr=*/1e-3f, /*beta1=*/0.9f, /*beta2=*/0.999f, /*eps=*/1e-8f,
#if WEIGHT_DECAY_ENABLE
    /*weight_decay=*/1e-2f,
#endif
    size,
    OptimizerType::ADAM,
    /*timestep=*/global_step,
    /*stream=*/0);
```

---

## 11) TODO

* Multi‑tensor fused optimizer (파라미터 다중 업데이트 1런치)
* 외부 워크스페이스 인터페이스로 글로벌 노름 클리핑 스트림‑세이프화
* Mixed Precision (FP16/BF16) 및 GradScaler 연동
* decoupled WD 계수 스케줄(코사인/step)
* 호스트‑측 리덕션 제거(디바이스 온리)
