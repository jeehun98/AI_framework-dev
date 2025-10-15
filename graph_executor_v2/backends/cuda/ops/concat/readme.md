N-차원 텐서들을 특정 축(`axis`)으로 이어 붙이는 연산자. 동일한 API로 **forward(Concat)** 와 **backward(Scatter-Add)** 를 제공한다.

```
📦 backends/cuda/ops/concat
 ┣ 📜 api.hpp        # 공용 API (Attrs/Launcher 프로토타입)
 ┣ 📜 launcher.cu    # shape/axis 검증, 커널 호출 시퀀스
 ┗ 📜 kernels.cu     # 범용 '영역 복사/가산' CUDA 커널

```

---

## 1) 개념 요약

### Forward (concat)

여러 입력 텐서 $X_0, X_1, \dots, X_{n-1}$를 축 `axis` 방향으로 이어 붙여 $Y$ 생성:

- 모든 $X_i$ 는 **rank 동일**, **row-major**, **FP32**, **CUDA 메모리**여야 함
- `axis` 를 제외한 모든 차원은 $Y$ 와 동일
- `axis` 차원 길이의 합이 $Y$의 해당 차원과 같아야 함

### Backward (scatter-add)

연쇄법칙에 의해 $\frac{\partial \mathcal{L}}{\partial X_i}$ 는 $\frac{\partial \mathcal{L}}{\partial Y}$의 `axis` 방향 슬라이스가 된다. 구현은 **영역 가산**(D += S) 방식으로 각 $gX_i$에 복원한다.

---

## 2) 지원/전제 (현재 구현 기준)

| 항목 | 상태 |
| --- | --- |
| DType | **FP32 전용** (`float`) |
| 레이아웃 | **Row-Major** |
| Rank | **1 ≤ rank ≤ 4** |
| 축 | `0 … rank-1` |
| 장치 | CUDA (device pointer) |
| 워크스페이스 | 없음 |
| 캡처 세이프 | ✅ (동적 할당 없음) |
| 결정성 | ✅ (원자연산/경합 없음) |

> 주의: Attrs는 int32 기반. 매우 큰 텐서에서 stride/dim의 32비트 한계를 넘는 경우는 현재 범위 밖.
> 

---

## 3) Public API (현재 시그니처 그대로)

### `api.hpp`

```cpp
struct ConcatAttrs {
  int rank{1};
  int axis{0};  // 0..rank-1
};

// Forward: Y = concat(Xs, axis)
Status ConcatCudaLaunch(const Tensor* Xs, int n,
                        Tensor& Y,
                        const ConcatAttrs& attrs,
                        StreamHandle stream);

// Backward: gX_i += slice(gY, axis, offset_i, size_i)
Status ConcatCudaBackwardLaunch(const Tensor& gY,
                                Tensor* gXs, int n,
                                const ConcatAttrs& attrs,
                                StreamHandle stream);

```

**계약/검증 (런처가 수행)**

- 모든 `Xs[i]` / `Y` 는 FP32, row-major, CUDA, rank 일치
- `axis` 제외 차원은 `Y`와 동일
- `sum_i Xs[i].shape[axis] == Y.shape[axis]`
- Backward는 `gXs[i]` 들이 `gY`와 `axis` 제외 차원 동일, `axis` 길이 합이 `gY.shape[axis]` 와 동일

---

## 4) 내부 동작: Launcher → Kernels

### 커널 공통 아이디어

`kernels.cu` 는 **범용 영역(region) 복사/가산** 커널 2개를 제공한다.

- `tensor_copy_region_kernel` : **Y[region] = X[region]**
- `tensor_add_region_kernel` : **D[region] += S[region]**

각 커널은:

- 최대 rank=4 까지 지원 (코드상 4원소 배열로 좌표/stride 처리)
- 블록당 256스레드, **그리드=총 원소 수 / 256**
- 총 원소 수/오프셋 계산은 **int64** 사용(오버플로 억제)
- 공유 메모리 사용 없음 → **Graph capture-safe**

### Forward 경로

1. `Y` 의 row-major stride 계산
2. 각 입력 `Xi` 에 대해:
    - `reg_dims = Xi.shape` (전 영역 복사)
    - `y_starts[axis] = concat_offset` 로 누적 위치 지정
    - `concat_copy_region_kernel_launcher()` 호출
        
        → 내부에서 `tensor_copy_region_kernel<<<grid, block>>>()`
        

### Backward 경로

1. `gY` 의 row-major stride 계산
2. 각 입력 그래디언트 `gXi` 에 대해:
    - `reg_dims = gXi.shape`
    - `s_starts[axis] = concat_offset` 로 `gY`에서 해당 구간 시작 지정
    - `concat_add_region_kernel_launcher()` 호출
        
        → 내부에서 `tensor_add_region_kernel<<<grid, block>>>()`
        
        → **`gXi += slice(gY)`** 형태(누적 연산)
        

> 중요: 현재 구현은 gXi에 += 로 기록한다. 일반적으로 concat의 역전파는 각 gXi 위치가 서로 겹치지 않기 때문에 결과적으로 = 와 동일하다. 다만, 호출 전 gXi가 0으로 초기화되어 있지 않다면 잔여값이 더해질 수 있으므로, 프레임워크 상에서 gXi 초기화를 보장하는 것이 안전하다.
> 

---

## 5) 커널 인터페이스(요약)

```cpp
// Forward 지역 복사
extern "C" void concat_copy_region_kernel_launcher(
  const float* X, float* Y, int rank,
  const int* reg_dims,          // [rank]
  const int* x_strides,         // [rank]
  const int* y_strides,         // [rank]
  const int* x_starts,          // [rank]
  const int* y_starts,          // [rank]
  cudaStream_t s);

// Backward 지역 가산
extern "C" void concat_add_region_kernel_launcher(
  const float* S, float* D, int rank,
  const int* reg_dims,          // [rank]
  const int* s_strides,         // [rank]
  const int* d_strides,         // [rank]
  const int* s_starts,          // [rank]
  const int* d_starts,          // [rank]
  cudaStream_t s);

```

**좌표/stride 체계**

- 모든 좌표/stride는 **row-major** 기준
- `reg_dims`: 처리할 **영역 크기**
- `_starts`: 원본/목적 텐서의 **시작 좌표**
- `_strides`: row-major stride(요소 단위)

---

## 6) 성능 관찰 포인트 (현재 구현 기준)

- **메모리 대역폭 지배**: 로직은 단순 산술+메모리 이동이므로, 대부분 **bandwidth-bound**.
- **coalescing**: `axis` 위치/stride 조합에 따라 접근 coalescing 정도가 달라짐.
    
    특히 `axis`가 **마지막 차원**이고 각 입력이 메모리상 **연속 블록**이면 유리.
    
- **벡터화 미사용**: 현재는 `float` 단위 로드/스토어. (현 상태 설명이므로 그대로 기록)

---

## 7) 예제 사용법

### Forward

```cpp
ai::Tensor X0, X1, Y;
ai::ConcatAttrs attrs{.rank=4, .axis=1}; // 예: NCHW에서 C축 concat

ai::Status st = ai::ConcatCudaLaunch(
  /*Xs=*/&X0, /*n=*/2, /*Y=*/Y,
  /*attrs=*/attrs, /*stream=*/stream
);
// X0, X1는 연속 메모리 및 row-major, FP32, CUDA여야 함

```

### Backward

```cpp
ai::Tensor gY, gX0, gX1;
ai::ConcatAttrs attrs{.rank=4, .axis=1};

ai::Status st = ai::ConcatCudaBackwardLaunch(
  /*gY=*/gY, /*gXs=*/&gX0, /*n=*/2,
  /*attrs=*/attrs, /*stream=*/stream
);
// 주의: gX0/gX1는 프레임워크에서 0으로 초기화되어 있거나,
// 이 호출이 유일한 작성 경로임이 보장되어야 함(+= 방식).

```

---

## 8) 에러/검증 규칙 (현재 반환 의미)

- 잘못된 포인터/빈 입력 목록 → `Status::Invalid`
- dtype/장치/레이아웃/rank 미일치 → `Status::Invalid`
- 축 범위 벗어남 → `Status::Invalid`
- 축 제외 차원 불일치 → `Status::ShapeMismatch`
- 축 길이 합 불일치 → `Status::ShapeMismatch`

*(커널 launch 이후의 CUDA 오류 확인은 현재 코드에 명시적 체크가 없다 — “현 상태 설명”)*

---

## 9) 캡처 세이프 & 결정성

- **동적 메모리 없음**, **공유 메모리 없음** → CUDA Graph 캡처 적합
- 각 스레드는 불변 매핑으로 독립 요소를 처리 → **결정적 결과**

---

## 10) 정리 (현재 상태)

- 단순하고 **일반화된 영역 복사/가산 커널**로 concat을 구성
- **FP32/row-major/최대 rank 4**에 특화
- Forward는 “연속 복사”, Backward는 “슬라이스 가산(+=)”
- **캡처-세이프**, **결정적**, **워크스페이스 불필요**
- 성능은 메모리 대역폭과 축/stride 배치에 좌우