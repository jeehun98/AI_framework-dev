from __future__ import annotations
from typing import List, Tuple, Any, Iterable, Optional
from .base import Layer

class Sequential(Layer):
    """
    단순 레이어 체인:
      - layers: [Layer, ...]
      - call(x): x를 순서대로 흘려보냄
      - backward(grad): 거꾸로 grad 전파
      - compute_output_shape: 레이어 체인으로 추론
      - summary(): 간단한 구조/shape 출력용 문자열 반환
    주의: 각 하위 Layer는 자신의 forward에서 역전파에 필요한 캐시를 내부에 저장한다고 가정.
    """
    def __init__(self, *layers: Layer, name: Optional[str] = None):
        super().__init__(name=name)
        self.layers: List[Layer] = list(layers)

    def add(self, layer: Layer) -> None:
        self.layers.append(layer)
        # 모델이 이미 build된 상태에서 add하면 shape 갱신 필요
        if self.built and self.output_shape is not None:
            ish = self.output_shape
            # 새 레이어를 즉시 build (가능하면)
            try:
                layer.build(ish)  # base.__call__의 자동 build 대신 명시 build
                osh = layer.compute_output_shape(ish)
            except Exception:
                # shape을 모르는 레이어면 패스 (런타임 첫 호출 때 build됨)
                osh = None
            if osh is not None:
                self.output_shape = tuple(map(int, osh))

    def build(self, input_shape: Tuple[int, ...]) -> None:
        cur = tuple(map(int, input_shape))
        for lyr in self.layers:
            # 레이어의 __call__이 자동 build를 하므로, 여기선 명시적으로 build 시도
            try:
                lyr.build(cur)
            except Exception:
                # 일부 레이어는 입력을 실제 텐서로 받아야 shape을 잡을 수 있음
                pass
            # compute_output_shape 시도
            try:
                cur = tuple(map(int, lyr.compute_output_shape(cur)))
            except Exception:
                # shape을 아직 모를 수 있음 -> 최초 실데이터 호출에서 확정
                pass
        self.input_shape = tuple(map(int, input_shape))
        self.output_shape = cur if isinstance(cur, tuple) else None
        self.built = True

    def call(self, x: Any):
        out = x
        for lyr in self.layers:
            out = lyr(out)
        return out

    def backward(self, grad_output: Any):
        """
        grad_output(= dLoss/dOut)을 마지막 레이어부터 처음까지 전파.
        보통 마지막이 손실이 아닌 경우: loss.backward가 반환한 dLast를 넣어주면 됨.
        """
        g = grad_output
        # 역순으로 backprop
        for lyr in reversed(self.layers):
            g = lyr.backward(g)
        return g

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        cur = tuple(map(int, input_shape))
        for lyr in self.layers:
            cur = lyr.compute_output_shape(cur)
        return cur

    def summary(self, indent: int = 2) -> str:
        lines = []
        pad = " " * indent
        lines.append(f"Sequential(name={self.name})")
        if self.input_shape:
            lines.append(f"{pad}Input:  {self.input_shape}")
        cur = self.input_shape
        for i, lyr in enumerate(self.layers):
            cls = lyr.__class__.__name__
            shp = None
            if cur is not None:
                try:
                    shp = lyr.compute_output_shape(cur)
                    cur = shp
                except Exception:
                    shp = "?"
                    cur = None
            lines.append(f"{pad}[{i:02d}] {cls:>20} -> {shp}")
        if cur is not None:
            lines.append(f"{pad}Output: {cur}")
        return "\n".join(lines)


# --------- (옵션) 아주 단순한 SGD 업데이트 유틸리티 ---------
def iter_params_with_grads(model: Sequential) -> Iterable[Tuple[Any, Any, str]]:
    """
    레이어에서 흔히 쓰는 파라미터 명명 규칙을 휴리스틱으로 스캔:
      - (W, dW), (weight, dweight), (b, db), (bias, dbias)
    레이어가 다른 이름을 쓰면, 그 레이어에 `parameters()` 메서드를 추가해
    [(param, grad, name), ...] 형태로 리턴하도록 확장하세요.
    """
    CANDIDATES = [
        ("W", "dW"),
        ("weight", "dweight"),
        ("b", "db"),
        ("bias", "dbias"),
    ]
    for lyr in model.layers:
        # 사용자 정의 parameters() 우선
        if hasattr(lyr, "parameters") and callable(getattr(lyr, "parameters")):
            for (p, g, n) in lyr.parameters():  # type: ignore
                yield (p, g, f"{lyr.__class__.__name__}.{n}")
            continue

        for p_name, g_name in CANDIDATES:
            if hasattr(lyr, p_name) and hasattr(lyr, g_name):
                p = getattr(lyr, p_name)
                g = getattr(lyr, g_name)
                yield (p, g, f"{lyr.__class__.__name__}.{p_name}")

def sgd_step(model: Sequential, lr: float) -> None:
    """
    매우 단순한 in-place SGD: p -= lr * g
    - 파라미터/그라드는 CuPy/Torch/NumPy 모두 브로드캐스트 가능한 형태면 동작.
    - 필요시 레이어별로 zero_grad()를 제공해 gradient 누적을 초기화하세요.
    """
    for p, g, name in iter_params_with_grads(model):
        if g is None:
            continue
        # CuPy/Torch/NumPy ndarray 모두 지원 (in-place 연산 가정)
        try:
            p[...] = p - lr * g
        except TypeError:
            # 파라미터가 텐서 객체일 때 .data로 접근해야 하는 경우 등
            if hasattr(p, "data"):
                p.data[...] = p.data - lr * g
            else:
                raise
