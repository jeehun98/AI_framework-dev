OPS_LAYERS_GUIDE.md
1. 공통 구조 개요

프로젝트는 크게 두 층으로 나뉩니다:

ops (저수준 연산 Wrappers)
→ C++/CUDA 바인딩을 감싸는 파이썬 래퍼.
→ tensor pointer, shape, stride 등을 직접 다루며 pybind11 모듈을 호출.

layers (고수준 Layer 추상화)
→ ops 모듈을 사용하여 실제 신경망 Layer를 구성.
→ 파라미터(가중치, 편향 등)와 초기화, forward/backward 로직을 포함.

2. ops 내부 구조
파일 위치
python/graph_executor_v2/ops/
 ┣ 📜__init__.py
 ┣ 📜common.py        # 텐서 헬퍼
 ┣ 📜gemm.py          # GEMM 연산 wrapper
 ┣ 📜conv2d.py        # Conv2D 연산 wrapper
 ┣ 📜pool2d.py        # Pool2D 연산 wrapper
 ┣ 📜softmax.py       # Softmax 연산 wrapper
 ┣ 📜cross_entropy.py # Cross-Entropy 연산 wrapper
 ┗ ...

공통 패턴

pybind11 모듈 불러오기

from graph_executor_v2.ops import require
g = require("gemm")   # → _ops_gemm


Tensor 생성기 사용

_ops_common에서 제공되는 make_tensor_2d, make_tensor_4d 등을 호출.

numpy/cupy 배열을 포인터 + shape로 변환 후 바인딩 호출.

Forward 함수 정의

def forward(A, B, bias=None, ...):
    tA = as_tensor_2d(A)
    tB = as_tensor_2d(B)
    tY = as_tensor_2d(Y)
    g.forward_ex(tA, tB, bias, tY, act="relu", with_bias=True)
    return Y


Backward 함수 정의 (선택)

필요 시 g.backward(...) 호출.

반환값: dX, dW, dB 등.

3. layers 내부 구조
파일 위치
python/graph_executor_v2/layers/
 ┣ 📜base.py        # Layer 추상 클래스
 ┣ 📜dense_gemm.py  # Dense layer
 ┣ 📜conv2d.py      # Conv2D layer
 ┣ 📜pool2d.py      # Pool2D layer
 ┣ 📜softmax_ce.py  # Softmax + CrossEntropy layer
 ┗ ...

공통 패턴

Layer 상속

from .base import Layer

class Dense(Layer):
    def __init__(self, units, activation="relu", initializer="he", use_native_bwd=True):
        super().__init__()
        ...


build()

입력 shape 기반으로 weight/bias 생성.

self.built = True

def build(self, input_shape):
    in_dim = input_shape[-1]
    self.W = cp.random.randn(in_dim, self.units).astype(cp.float32)
    self.b = cp.zeros(self.units, dtype=cp.float32)
    self.built = True


call()

forward 연산 정의.

내부적으로 ops 모듈을 호출.

def call(self, x):
    return gemm_ops.forward(x, self.W, self.b, act=self.activation)


backward()

필요 시 ops.backward 사용.

4. 바인딩 → ops → layer 작성 순서

바인딩 코드 작성 (src/bindings/..._pybind.cpp)

make_tensor_* 유틸 구현.

forward/backward 바인딩 등록.

ops Python wrapper 작성 (python/graph_executor_v2/ops/...py)

require("_ops_xxx") 로 모듈 불러오기.

Tensor 변환 후 바인딩 함수 호출.

layer Python 구현 (python/graph_executor_v2/layers/...py)

Layer 상속.

build()에서 파라미터 초기화.

call()에서 ops.forward 호출.

5. 예시: Conv2D
바인딩
m.def("forward", [](uintptr_t x_ptr, ..., Conv2DAttrs attrs, uintptr_t stream) {
    auto st = Conv2DCudaLaunch(X, W, B, Y, attrs, stream);
    throw_if_bad(st, "Conv2DCudaLaunch");
});

ops/conv2d.py
from . import require
_g = require("conv2d")

def forward(X, W, b=None, attrs=None):
    tX = as_tensor_4d(X)
    tW = as_tensor_4d(W)
    tY = make_empty_output(...)
    _g.forward(int(X.data.ptr), list(X.shape),
               int(W.data.ptr), list(W.shape),
               int(tY.data.ptr), list(tY.shape),
               b_ptr if b is not None else None,
               attrs or _g.Conv2DAttrs())
    return tY

layers/conv2d.py
from .base import Layer
import graph_executor_v2.ops.conv2d as conv_ops

class Conv2D(Layer):
    def build(self, input_shape):
        N, C, H, W = input_shape
        self.W = cp.random.randn(self.out_channels, C, kh, kw).astype(cp.float32)
        self.b = cp.zeros(self.out_channels, dtype=cp.float32)
        self.built = True

    def call(self, x):
        return conv_ops.forward(x, self.W, self.b, self.attrs)


👉 이 문서만 있으면, 새로운 연산을 추가할 때도

C++ 바인딩 → ops 래퍼 → layer 구현 순서로 바로 만들 수 있습니다.