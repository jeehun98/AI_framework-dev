# graph_executor_v2/layers/sequential.py
from __future__ import annotations
from typing import List, Tuple, Any, Iterable, Optional

from .base import Layer

CANDIDATE_PARAM_GRAD_NAMES = [
    ("W", "dW"),
    ("weight", "dweight"),
    ("b", "db"),
    ("bias", "dbias"),
]

class Sequential(Layer):
    """
    - layers: [Layer, ...]
    - call(x): 순서대로 forward
    - backward(grad): 역순 전파
    - parameters(): (param, grad, name) 이터레이터
    - zero_grad(): grad 초기화
    - state_dict()/load_state_dict(): 저장/로드
    - train()/eval(): 서브 레이어까지 모드 전파
    - attach_grads(): 레이어 내부 dW/db 등을 param.grad로 연결
    """
    def __init__(self, *layers: Layer, name: Optional[str] = None):
        super().__init__(name=name)
        self.layers: List[Layer] = list(layers)
        self.training: bool = True  # 기본 학습 모드

    def add(self, layer: Layer) -> None:
        self.layers.append(layer)
        if self.built and self.output_shape is not None:
            ish = self.output_shape
            try:
                layer.build(ish)
                osh = layer.compute_output_shape(ish)
            except Exception:
                osh = None
            if osh is not None:
                self.output_shape = tuple(map(int, osh))

    def build(self, input_shape: Tuple[int, ...]) -> None:
        cur = tuple(map(int, input_shape))
        for lyr in self.layers:
            try:
                lyr.build(cur)
            except Exception:
                pass
            try:
                cur = tuple(map(int, lyr.compute_output_shape(cur)))
            except Exception:
                pass
        self.input_shape = tuple(map(int, input_shape))
        self.output_shape = cur if isinstance(cur, tuple) else None
        self.built = True

    # === Runtime ===
    def call(self, x: Any):
        out = x
        for lyr in self.layers:
            # (선택) 드롭아웃/BN 등은 lyr.training 플래그를 참조하도록 구현
            if hasattr(lyr, "training"):
                lyr.training = self.training
            out = lyr(out)
        return out

    def backward(self, grad_output: Any):
        g = grad_output
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
        total_params = 0
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
            # 파라미터 수 집계(가능한 경우)
            pcount = 0
            try:
                for (p, _, _) in lyr.parameters():  # type: ignore
                    try:
                        pcount += int(p.size) if hasattr(p, "size") else int(p.size())
                    except Exception:
                        pass
            except Exception:
                pass
            total_params += pcount
            lines.append(f"{pad}[{i:02d}] {cls:>20} -> {shp}  (params={pcount})")
        if cur is not None:
            lines.append(f"{pad}Output: {cur}")
        lines.append(f"{pad}Total params: {total_params}")
        return "\n".join(lines)

    # === Training helpers ===
    def train(self, mode: bool = True):
        """model.train()/eval() 호환"""
        self.training = bool(mode)
        for lyr in self.layers:
            if hasattr(lyr, "training"):
                lyr.training = self.training
        return self

    def eval(self):
        return self.train(False)

    # 표준화된 파라미터 인터페이스
    def parameters(self) -> Iterable[Tuple[Any, Any, str]]:
        for lyr in self.layers:
            if hasattr(lyr, "parameters") and callable(getattr(lyr, "parameters")):
                # (param, grad, name) or (param, grad)
                for t in lyr.parameters():  # type: ignore
                    if len(t) == 3:
                        yield t  # (p, g, name)
                    else:
                        p, g = t
                        yield (p, g, lyr.__class__.__name__)
                continue
            # 휴리스틱 스캔 (W/dW, b/db 등)
            for p_name, g_name in CANDIDATE_PARAM_GRAD_NAMES:
                if hasattr(lyr, p_name) and hasattr(lyr, g_name):
                    p = getattr(lyr, p_name)
                    g = getattr(lyr, g_name)
                    yield (p, g, f"{lyr.__class__.__name__}.{p_name}")

    def zero_grad(self):
        """param.grad 및 레이어 내부 dW/db 등을 0으로."""
        # param.grad 초기화
        for (p, _, _) in self.parameters():
            g = getattr(p, "grad", None)
            if g is not None:
                try:
                    g[...] = 0
                except Exception:
                    if hasattr(g, "zero_"):
                        g.zero_()
                    else:
                        setattr(p, "grad", None)
        # 레이어 내부 그라드 캐시 초기화
        for lyr in self.layers:
            # 사용자 정의 zero_grad가 있으면 우선
            if hasattr(lyr, "zero_grad") and callable(getattr(lyr, "zero_grad")):
                try: lyr.zero_grad()  # type: ignore
                except Exception: pass
                continue
            # 없으면 휴리스틱
            for _, g_name in CANDIDATE_PARAM_GRAD_NAMES:
                if hasattr(lyr, g_name):
                    g = getattr(lyr, g_name)
                    try:
                        g[...] = 0
                    except Exception:
                        try:
                            if hasattr(g, "zero_"): g.zero_()
                            else: setattr(lyr, g_name, None)
                        except Exception:
                            pass

    def attach_grads(self):
        """
        레이어 내부 dW/db 등을 param.grad에 연결.
        - Optimizer가 param.grad를 읽는 구조(AdamW/SGD 등) 지원.
        - 역전파 직후 호출.
        """
        for (p, g, _) in self.parameters():
            if g is not None:
                setattr(p, "grad", g)

    # === 저장/로드 ===
    def state_dict(self) -> dict:
        """
        레이어가 state_dict() 제공하면 위임.
        없으면 (W,b 등) 파라미터를 추출해 저장.
        """
        sd = {"name": self.name, "layers": []}
        for lyr in self.layers:
            entry = {"class": lyr.__class__.__name__}
            if hasattr(lyr, "state_dict"):
                try:
                    entry["state"] = lyr.state_dict()  # type: ignore
                except Exception:
                    entry["state"] = None
            else:
                # 휴리스틱 파라미터 스냅샷
                params = {}
                for p_name, _ in CANDIDATE_PARAM_GRAD_NAMES:
                    if hasattr(lyr, p_name):
                        params[p_name] = getattr(lyr, p_name)
                entry["params"] = params
            sd["layers"].append(entry)
        return sd

    def load_state_dict(self, sd: dict):
        """
        저장한 순서/타입이 동일하다는 가정.
        """
        assert len(sd.get("layers", [])) == len(self.layers), "layer count mismatch"
        for lyr, entry in zip(self.layers, sd["layers"]):
            if hasattr(lyr, "load_state_dict") and entry.get("state", None) is not None:
                try:
                    lyr.load_state_dict(entry["state"])  # type: ignore
                    continue
                except Exception:
                    pass
            # 휴리스틱 로드
            params = entry.get("params", {})
            for k, v in params.items():
                if hasattr(lyr, k):
                    try:
                        getattr(lyr, k)[...] = v
                    except Exception:
                        setattr(lyr, k, v)
        return self
