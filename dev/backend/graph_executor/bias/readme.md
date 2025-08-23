# bias — Row/Col-wise Bias Add Kernels

본 모듈은 per‑sample 행렬 표현(`rows × cols`)에 대해 **행(열) 방향 브로드캐스트 bias add**를 제공합니다.
`add_bias_rowwise.cuh`에는 커널과 런처가 함께 정의되어 있습니다.

---

## 1. 목적/책임

* Dense/Activation 직후 등에서 **열 방향(row‑wise) bias**를 더하는 연산.
* CNN 등에서 채널 단위로 **행 방향(col‑wise) bias**를 브로드캐스트하는 연산.
* 단순 합 연산이므로 **in‑place** 사용 가능(`out==in`).

---

## 2. 퍼블릭 API (런처)

```cpp
void launch_add_bias_rowwise(const float* in, const float* bias, float* out,
                             int rows, int cols, cudaStream_t stream);

void launch_add_bias_colwise(const float* in, const float* bias, float* out,
                             int rows, int cols, int rows_per_sample,
                             cudaStream_t stream);
```

* `rows, cols`는 **per‑sample shape** 기준의 전체 연산 크기에서 `rows = batch_size * rows_per_sample`입니다.
* `rowwise`: `bias` 길이는 **`cols`** 여야 합니다. 열(특징)마다 다른 상수 더하기.
* `colwise`: `bias` 길이는 **`rows_per_sample`** 여야 합니다. 샘플 내 행(채널/시퀀스 위치)마다 상수 더하기.

---

## 3. 커널 요약

```cpp
__global__ void add_bias_rowwise_kernel(const float* in, const float* bias,
                                        float* out, int rows, int cols) {
    // out[r,c] = in[r,c] + bias[c]
}

__global__ void add_bias_colwise_kernel(const float* in, const float* bias,
                                        float* out, int rows, int cols, int rows_per_sample) {
    // out[r,c] = in[r,c] + bias[r % rows_per_sample]
}
```

* 그리드 구성: 기본 `block=(32,8)`, `grid=(ceil(cols/32), ceil(rows/8))`.
* `CUDA_CHECK(cudaGetLastError())`로 런처에서 직후 오류 확인.

---

## 4. Shape/브로드캐스트 규약

* **Per‑Sample Shape 규칙**과 정합:

  * 실행기에서 `B = batch_size * rows_per_sample`을 계산해 `rows`로 전달합니다.
* **row‑wise**: `(B, C)` + `(1, C)` → `(B, C)`
* **col‑wise**: `(B, C)` + `(rows_per_sample, 1)` → `(B, C)` (실제로는 길이 `rows_per_sample` 벡터를 사용)
* CNN에서 채널 편향을 `col‑wise`로 쓰는 경우, `rows_per_sample = channels`(또는 내부 표현상의 행 수).

---

## 5. 메모리/레이아웃/성능

* 포인터는 모두 **float32, C‑contiguous** 가정.
* x축=열(col)로 매핑하여 **coalesced read/write**.
* **In‑place 지원**: `out`을 `in`과 동일 포인터로 넘겨도 안전.
* 튜닝 아이디어:

  * `float2/float4` 벡터화(열 길이가 2/4 배수일 때)
  * 큰 `cols`에서는 `blockDim.x=128/256`로 조정 후 occupancy 측정
  * L2 hit 개선을 위해 bias를 `__ldg` 또는 shared에 캐시(현대 GPU에선 자동 캐시로 충분한 경우 多)

---

## 6. 실행기(executor) 연동 예시

```cpp
// ADD op를 bias‑전용으로 사용할 때 (예: param_id가 bias)
case ge::OpType::ADD: {
    const auto& in_id  = op.input_id;
    const auto& out_id = op.output_id;
    const auto& b_id   = op.param_id; // bias

    float* x  = tensors.at(in_id);
    float* y  = tensors.at(out_id);
    const float* b = tensors.at(b_id);

    const ge::Shape& shp = shapes.at(out_id);
    int rows = shp.rows; // per‑sample rows
    int cols = shp.cols;

    // rowwise 기본: bias 길이 == cols
    launch_add_bias_rowwise(x, b, y, /*rows=*/batch_size*rows, cols, 0);
    break;
}
```

> **주의:** CNN 내부 표현이 `(rows=channels, cols=H_out*W_out)`인 경우 `col‑wise` 경로가 맞습니다. 이때 `rows_per_sample=channels`를 전달해야 합니다.

---

## 7. BWD(그라디언트) 설계 메모

* `∂L/∂x = ∂L/∂y` (단순 복사)
* `∂L/∂bias`는 **축소(리듀스)** 가 필요:

  * row‑wise: `db[c] = sum_r dY[r,c]` → `reduce/rowwise_sum` 커널 사용 권장
  * col‑wise: `db[r_local] = sum_{samples, c} dY[r,c]` where `r_local = r % rows_per_sample`

    * 구현 시 `atomicAdd` 또는 두 단계 리듀스(성능 우수)
* 현재 레포 구조상 **bias grad 커널은 reduce/ 폴더**에 배치하는 것을 권장합니다.

---

## 8. 수치/안전 가이드

* 입력/출력/바이어스가 NaN/Inf면 결과가 전파됩니다. 필요 시 상위 레이어에서 NaN 가드 권장.
* `rows_per_sample`은 0이 될 수 없고, `rows % rows_per_sample == 0`이어야 합니다(런처에서 assert 고려).

---

## 9. 간단 스모크 테스트

```cpp
int B=4, rows_ps=1, C=8; // per‑sample rows=1 예시
std::vector<float> x(B*C, 1.0f), b(C); // b=[0,1,2,...]
std::iota(b.begin(), b.end(), 0.f);
// GPU 업로드 후 launch_add_bias_rowwise(...)
// 기대: y[r,c] = 1 + c
```

파이썬/쿠피 연동 예시는 Dense→Bias(ADD)→Activation 체인을 이용해 `evaluate/fit`가 NaN 없이 동작하는지 확인합니다.

---

## 10. TODO

* `float4` 벡터화 경로 추가
* fused epilogue: GEMM/Conv 결과에 bias+activation을 한 커널로 결합
* half/bfloat16 mixed precision 경로
* col‑wise bias의 고성능 리듀스 커널(grad) 제공
