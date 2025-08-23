# activation — CUDA 활성화 커널 모듈

본 문서는 `activation/` 폴더의 책임, 퍼블릭 API, Shape 규약, 수치 안정성, 디스패처 연동, 테스트 가이드를 정리합니다. 현재 구현은 `activation_ops.cu/.cuh`로 구성되어 있습니다.

---

## 1. 목적/책임

* ReLU, Sigmoid, Tanh, LeakyReLU, ELU, GELU(tanh 근사), SiLU(Swish) 등의 **pointwise 활성화** 연산을 제공합니다.
* **FWD/BWD** 커널을 노출하고, 필요 시 **bias add**를 FWD 경로에서 함께 처리합니다(옵션 인자 `bias`).

---

## 2. 지원 Op 식별자 (내부 enum)

```cpp
// activation_ops.cuh
enum {
  ACT_IDENTITY = 0,
  ACT_RELU     = 1,
  ACT_SIGMOID  = 2,
  ACT_TANH     = 3,
  ACT_LEAKY    = 4,
  ACT_ELU      = 5,
  ACT_GELU     = 6,
  ACT_SILU     = 7
};
```

> **주의:** 이는 **내부 활성화 식별자**입니다. Graph의 `OpType` 값(예: `RELU=2, SIGMOID=3, TANH=4` 등)과 **다를 수 있으므로** 실행기(`executor/`)의 디스패처에서 매핑해야 합니다.

예) 디스패처 매핑 테이블(권장):

```cpp
int to_act_id(ge::OpType t) {
    switch (t) {
        case ge::OpType::RELU:    return ACT_RELU;
        case ge::OpType::SIGMOID: return ACT_SIGMOID;
        case ge::OpType::TANH:    return ACT_TANH;
        case ge::OpType::LEAKY:   return ACT_LEAKY;   // 있으면
        case ge::OpType::ELU:     return ACT_ELU;     // 있으면
        case ge::OpType::GELU:    return ACT_GELU;    // 있으면
        case ge::OpType::SILU:    return ACT_SILU;    // 있으면
        default:                  return ACT_IDENTITY;
    }
}
```

---

## 3. 퍼블릭 API (런처)

```cpp
void launch_activation_forward(const float* in, const float* bias, float* out,
                               int rows, int cols, int act_type,
                               float alpha, int gelu_tanh_flag,
                               cudaStream_t stream);

void launch_activation_backward(const float* grad_out,
                                const float* in,      // pre-activation z
                                const float* out,     // f(z)
                                float* grad_in,
                                int rows, int cols, int act_type,
                                float alpha, int gelu_tanh_flag,
                                cudaStream_t stream);
```

* `bias`는 **선택**(null 허용). 크기는 `cols`(per-sample 열 수)와 동일.
* `alpha`는 Leaky/ELU에 사용. 그 외 활성화에서는 무시.
* `gelu_tanh_flag`: 1=tanh 근사, 0=추후 precise(ERF) 분기 등 확장용.
* 그리드 구성은 `(block= {ACT_BLOCK_X, ACT_BLOCK_Y}, grid={(cols+bx-1)/bx, (rows+by-1)/by})`.

---

## 4. Shape 규약 (Per‑Sample)

* 입력/출력은 **per‑sample** 행렬(`rows × cols`)로 해석합니다.
* 전체 연산 요소 수: `B*C`, 여기서

  * `C = cols`,
  * `B = batch_size * rows` (시퀀스/채널 행을 rows로 둘 수 있음)
* **Bias**는 `(1, C)`를 가정하고 열 방향으로 브로드캐스트합니다.

---

## 5. 수치 안정성

* Sigmoid: 오버/언더플로우 방지를 위한 안정화 구현(`sigmoid_stable`). 출력은 `[eps, 1-eps]`로 클램프.
* Tanh/ELU: `isfinite` 가드 및 합리적 클램프.
* SiLU: `y = x * sigmoid(x)`, 도함수 `s + x*s*(1-s)` 사용.
* GELU(tanh 근사): `0.5*x*(1+tanh(√(2/π)*(x+0.044715x^3)))`. BWD는 체인룰 기반.
* NaN/Inf 감지 시 0으로 대체하고 선택적 로깅(`KPRINTF`) 수행.

---

## 6. 메모리/레이아웃/성능

* **Coalesced 접근**: x축을 열(col)에, y축을 행(row)에 매핑(기본 블록: 32×8).
* 모든 포인터는 **C‑contiguous float32**를 가정합니다.
* **In‑place 지원**: `out==in` 가능(FWD). 스레드가 같은 인덱스를 읽고 동일 인덱스에 쓰므로 안전.

  * BWD에서 `grad_in==grad_out`은 권장하지 않음(명시적 분리 권장).
* 향후 최적화: `float2/float4` 벡터화(열이 2/4 정렬), shared memory 타일링, `-use_fast_math` 플래그 검토.

---

## 7. 실행기(executor) 연동 (예시)

```cpp
// run_graph.cu 내부 디스패치 예시
case ge::OpType::SIGMOID:
case ge::OpType::RELU:
case ge::OpType::TANH:
case ge::OpType::GELU: {
    const std::string& in_id  = op.input_id;
    const std::string& out_id = op.output_id;
    const std::string& b_id   = op.param_id; // bias가 ADD로 분리되어 있으면 빈 문자열일 수 있음

    auto t_in  = tensors.at(in_id);
    auto t_out = tensors.at(out_id);
    const float* bias = nullptr;
    if (!b_id.empty() && tensors.count(b_id)) bias = tensors.at(b_id);

    const ge::Shape& shp = shapes.at(out_id); // per-sample
    int rows = shp.rows, cols = shp.cols;

    int act = to_act_id(op.op_type);
    float alpha = op.extra.alpha;          // Leaky/ELU용 (없으면 0)
    int gelu_tanh = 1;                     // 현재 구현은 tanh 근사 고정

    launch_activation_forward(
        t_in, bias, t_out, rows, cols, act, alpha, gelu_tanh, /*stream=*/0);
    break;
}
```

> **주의:** Bias가 ADD op로 분리된 모델이라면 `param_id`가 비어있을 수 있으며, 그 경우 `bias=nullptr`로 전달합니다.

BWD 연동 시에는 `grad_out`, `in(z)`, `out(f(z))` 포인터가 모두 필요합니다.

* `grad_out`는 다음 op에서 전파된 그라디언트, `grad_in`은 현재 op 앞단으로 전달할 버퍼입니다.

---

## 8. 미분 공식 요약

* ReLU: `d/dz max(z,0) = 1[z>0]`
* Sigmoid: `σ'(z) = σ(z)(1-σ(z))`
* Tanh: `1 - tanh(z)^2`
* LeakyReLU(α): `z>=0?1:α`
* ELU(α): `z>=0?1:out+α` (out=α(e^z−1))
* SiLU: `s + z*s*(1-s)` where `s=σ(z)`
* GELU(tanh 근사): 문서화된 근사 도함수 사용(코드 참고)

---

## 9. 테스트 가이드

* **스모크(FWD)**: 각 활성화에 대해 난수 입력 `(rows=1, cols=C)`로 실행, NaN/Inf가 없는지 확인.
* **수치 미분(BWD)**:

  1. 스칼라 손실 `L = sum(out)` 설정
  2. `dL/dz`를 수치 미분과 비교(ε=1e-3)
* **그래프 연동**: Dense→Activation 체인으로 `evaluate/fit`가 NaN 없이 동작하는지.

간단 파이썬 예시:

```python
x = np.random.randn(8, 16).astype(np.float32)
# GE 래퍼로 in/out 바인딩 후 launch_activation_forward 호출 → out shape=(8,16)
```

---

## 10. 로깅/디버깅

* `logging_config.h`가 있으면 `KPRINTF`를 통해 커널 경고 1회 출력(`idx==0` 스레드 제한).
* `CUDA_LAUNCH_BLOCKING=1`으로 첫 오류 즉시 표면화.
* `GE_DEBUG_SYNC`가 켜진 경우, 각 디스패치 후 `cudaDeviceSynchronize()` 호출.

---

## 11. TODO

* ERF 기반 **정확한 GELU** 분기 추가(`gelu_tanh_flag=0`)
* 벡터화(float2/4), half/bfloat16 mixed precision 경로
* in‑place BWD 검증 및 최적화
* bias‑fuse를 activation 내부에서 옵션화 (현재는 `bias` 인자로 대체)
