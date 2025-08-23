# cnn — Conv2D CUDA Kernels (NCHW)

본 모듈은 **NCHW** 레이아웃에서 동작하는 Conv2D의 **Forward / Backward(dX, dW)** 커널과 런처를 제공합니다. (Bias/Activation은 별도 모듈)

---

## 1. 목적/책임

* Forward: `Y = conv2d(X, W; stride, padding)`
* Backward‑Input: `dX = dY * W^T` (수학적으로 correlation)
* Backward‑Weight: `dW = X ⊗ dY` (입력/출력 위치에 대한 합)
* 데이터 레이아웃: **NCHW**

  * `X: [N, Cin, Hin, Win]`
  * `W: [Cout, Cin, Kh, Kw]`
  * `Y: [N, Cout, Hout, Wout]`

> Bias 추가와 활성화(예: ReLU, GELU)는 **bias/**, **activation/** 모듈을 사용해 별도 연산으로 연결합니다. (fuse는 이후 TODO)

---

## 2. 퍼블릭 API (런처)

```cpp
void launch_conv2d_forward_nchw(
    const float* X, const float* W, float* Y,
    int N, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw,
    cudaStream_t stream);

void launch_conv2d_backward_input_nchw(
    const float* dY, const float* W, float* dX,
    int N, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw,
    cudaStream_t stream);

void launch_conv2d_backward_weight_nchw(
    const float* dY, const float* X, float* dW,
    int N, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw,
    cudaStream_t stream);
```

* 블록 크기: `TW=16` → `block=(16,16,1)` (dW는 `block=(Kw,1,1)`)
* 그리드:

  * FWD: `grid=((Wout+15)/16, (Hout+15)/16, N*Cout)`
  * dX : `grid=((Win +15)/16, (Hin +15)/16, N*Cin)`
  * dW : `grid=(Kh, Cin, Cout)` (스레드 x축이 `kw`)

---

## 3. 출력 크기 공식

* `Hout = floor((Hin + 2*Ph - Kh)/Sh) + 1`
* `Wout = floor((Win + 2*Pw - Kw)/Sw) + 1`
* **SAME** 패딩의 일반적 가정: `Ph = ⌊Kh/2⌋`, `Pw = ⌊Kw/2⌋` (대칭)

---

## 4. 인덱싱(핵심 포인트)

* Forward: `(n, oc, oh, ow)`에 대해 `ic, kh, kw` 합산

  * 입력 좌표: `h = oh*Sh - Ph + kh`, `w = ow*Sw - Pw + kw`
  * 경계 밖은 건너뜀(`unsigned` 비교)
* dX: `(n, ic, h, w)`에 대해 `oc, kh, kw` 합산

  * `oh = (h + Ph - kh)`, `ow = (w + Pw - kw)`
  * stride 정합: `if (oh % Sh != 0) continue;` → `oh/=Sh` 후 범위 검사
* dW: `(oc, ic, kh, kw)`마다 `n, oh, ow` 합산

> 음수 인덱스 처리: dX에서 `oh/ow`가 음수일 수 있으므로 `%`/`/` 사용 후 **범위 검사 전에** 조기 continue. 현재 커널은 그 패턴을 따릅니다.

---

## 5. 프레임워크 연결 (Per‑Sample Shape 규칙)

* 내부 표현은 Conv 출력이 `(filters, Hout*Wout)` 행렬일 수 있으나,
* **공식 출력**은 보통 **Flatten 레이어**로 `(1, filters*Hout*Wout)`로 변환하여 Dense와 연결합니다.
* `to_e_matrix()`에서 Shape 등록 예:

  * `input_id  → Shape(1, Cin*Hin*Win)`
  * `conv_out  → Shape(filters, Hout*Wout)`
  * `output_id → Shape(filters, Hout*Wout)` (Conv 자체는 그대로 내보내고, 별도 Flatten에서 `(1, FHW)`)
* Bias는 `(1, filters)` 또는 타일된 `(1, F*Hout*Wout)` (브로드캐스트 지원 여부에 따라 선택).

---

## 6. 수치/안전

* 포인터는 모두 **float32, C‑contiguous**를 가정합니다.
* OOB 접근은 모두 조건문으로 차단.
* 커널 런치 후 `cudaGetLastError()` 체크 권장.

---

## 7. 성능/튜닝 메모

* 현재 커널은 **레퍼런스(직접 합산)** 형태입니다.
* 향후 최적화 아이디어:

  1. 공유 메모리 타일링(입력/필터 tile)
  2. `im2col + GEMM` 경로(cuBLAS 활용)
  3. weight‑gradient(dW)에서 병렬화 재설계(현재는 파라미터별 단일 스레드 축소)
  4. 스트라이드/패딩에 따른 branch 수를 줄이도록 루프 변형
  5. float2/float4 vectorized load, L2 prefetch
  6. NHWC 경로 및 TensorCore 지원(fp16/bf16) 분기

---

## 8. 실행기(executor) 연동 예시

```cpp
// Forward
launch_conv2d_forward_nchw(
    X, W, Y,
    N, Hin, Win, Cin,
    Hout, Wout, Cout,
    Kh, Kw, Sh, Sw, Ph, Pw,
    /*stream=*/0);

// Backward
launch_conv2d_backward_input_nchw(dY, W, dX, N, Hin, Win, Cin,
                                  Hout, Wout, Cout, Kh, Kw, Sh, Sw, Ph, Pw, 0);
launch_conv2d_backward_weight_nchw(dY, X, dW, N, Hin, Win, Cin,
                                   Hout, Wout, Cout, Kh, Kw, Sh, Sw, Ph, Pw, 0);
```

---

## 9. 테스트 가이드

* 스모크(FWD): 난수 `X, W`로 FWD만 실행 후 `Y` NaN/Inf 없는지 확인, shape 검증.
* 수치 미분(BWD): 작은 텐서 크기에서 `dX/dW`를 수치 미분과 비교(ε=1e‑3).
* 통합: Conv→Bias→Activation→Flatten→Dense 체인으로 `evaluate/fit` 안정성 확인.

---

## 10. TODO

* Conv+bias+activation **fused epilogue**
* Groups/Depthwise Conv
* Padding SAME/VALID 자동 계산 유틸
* Mixed‑precision(fp16/bf16) & TensorCore 경로
* NHWC 변형 커널
