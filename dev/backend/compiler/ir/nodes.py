# compiler/ir/nodes.py
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Sequence, Tuple, Optional, Iterable
import itertools

# -----------------------------------------------------------------------------
# 간단 유틸: 이름 생성기
# -----------------------------------------------------------------------------
class _NameSeq:
    def __init__(self):
        self._counters: Dict[str, itertools.count] = {}

    def make(self, prefix: str) -> str:
        c = self._counters.setdefault(prefix, itertools.count(0))
        return f"{prefix}_{next(c)}"

_name_seq = _NameSeq()

def _as_shape(shape: Sequence[int] | Iterable[int]) -> Tuple[int, ...]:
    t = tuple(int(s) for s in shape)
    if any(s <= 0 for s in t):
        raise ValueError(f"invalid shape (must be >0): {t}")
    return t

# 허용 값 간단 체크(가벼운 방어막)
_ALLOWED_DTYPES = {"f16", "f32", "i8", "i32"}
_ALLOWED_LAYOUTS = {"rowmajor", "NCHW", "NHWC"}
_ALLOWED_DEVICES = {"cuda", "cpu"}

def _check_choice(name: str, val: str, allowed: set[str]) -> str:
    if val not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)} (got {val!r})")
    return val

# -----------------------------------------------------------------------------
# Tensor
# -----------------------------------------------------------------------------
@dataclass
class Tensor:
    # name을 비워두면 자동 생성
    name: str | None = None
    shape: Tuple[int, ...] = field(default_factory=lambda: (1,))
    dtype: str = "f32"          # "f16" | "f32" | "i8" | "i32"
    layout: str = "rowmajor"    # "rowmajor" | "NCHW" | "NHWC"
    device: str = "cuda"        # "cuda" | "cpu"

    # 실행기에서 디바이스 포인터 추출할 때 쓰는 *옵션* 실데이터 핸들(CuPy/Torch 등)
    t: Any | None = None

    def __post_init__(self):
        if self.name is None:
            self.name = _name_seq.make("t")
        self.shape = _as_shape(self.shape)
        self.dtype = _check_choice("dtype", self.dtype, _ALLOWED_DTYPES)
        self.layout = _check_choice("layout", self.layout, _ALLOWED_LAYOUTS)
        self.device = _check_choice("device", self.device, _ALLOWED_DEVICES)

    @property
    def rank(self) -> int:
        return len(self.shape)

    def with_data(self, data: Any) -> "Tensor":
        """실제 디바이스 배열을 붙여 반환(불변 스타일 편의)."""
        return replace(self, t=data)

# -----------------------------------------------------------------------------
# Op
# -----------------------------------------------------------------------------
@dataclass
class Op:
    op_type: str
    inputs: List[Tensor] = field(default_factory=list)
    outputs: List[Tensor] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    name: str | None = None

    def __post_init__(self):
        if self.name is None:
            self.name = _name_seq.make(self.op_type.lower())

    def add_input(self, t: Tensor) -> None:
        self.inputs.append(t)

    def add_output(self, t: Tensor) -> None:
        self.outputs.append(t)

# -----------------------------------------------------------------------------
# Graph
# -----------------------------------------------------------------------------
@dataclass
class Graph:
    # ops를 지정하지 않으면 빈 리스트로 시작
    ops: List[Op] = field(default_factory=list)
    name: str | None = None

    def __post_init__(self):
        if self.name is None:
            self.name = _name_seq.make("g")

    # 편의: 연산 추가 + 반환
    def add_op(self, op_type: str, *,
               inputs: Sequence[Tensor] = (),
               outputs: Sequence[Tensor] = (),
               attrs: Optional[Dict[str, Any]] = None,
               name: str | None = None) -> Op:
        op = Op(
            op_type=op_type,
            inputs=list(inputs),
            outputs=list(outputs),
            attrs=dict(attrs or {}),
            name=name,
        )
        self.ops.append(op)
        return op

    # 검증: 이름 유일성, 참조 무결성
    def validate(self) -> None:
        seen_names: set[str] = set()
        # 텐서 이름 중복 체크
        for t in self.tensors():
            if t.name in seen_names:
                raise ValueError(f"duplicate tensor name: {t.name!r}")
            seen_names.add(t.name)

        # Op의 inputs/outputs가 Tensor인지 체크
        for op in self.ops:
            if not all(isinstance(x, Tensor) for x in op.inputs):
                raise TypeError(f"{op.name}: inputs must be Tensor")
            if not all(isinstance(x, Tensor) for x in op.outputs):
                raise TypeError(f"{op.name}: outputs must be Tensor")

    # 그래프 내 모든 텐서 열거(순서 보장 X)
    def tensors(self) -> List[Tensor]:
        ts: List[Tensor] = []
        for op in self.ops:
            ts.extend(op.inputs)
            ts.extend(op.outputs)
        return ts

    # 얕은 복제(ops 리스트만 복사; Tensor/Op 객체는 동일 참조)
    def clone_shallow(self) -> "Graph":
        return Graph(ops=list(self.ops), name=f"{self.name}_clone")

    # 간단 토폴로지 정렬 (의존성 정보를 별도로 안 갖고 있어, 현재 순서를 유지)
    def topo_ops(self) -> List[Op]:
        return list(self.ops)

# -----------------------------------------------------------------------------
# 헬퍼: 간단 GEMM_BIAS_ACT 그래프 만들기
# -----------------------------------------------------------------------------
def make_gemm_bias_act_graph(M: int, N: int, K: int,
                             dtype_in: str = "f16",
                             dtype_out: str = "f16",
                             has_bias: bool = True,
                             act: str = "relu") -> Tuple[Graph, Tensor, Tensor, Optional[Tensor], Tensor]:
    A = Tensor(shape=(M, K), dtype=dtype_in, device="cuda")
    B = Tensor(shape=(K, N), dtype=dtype_in, device="cuda")
    C = Tensor(shape=(M, N), dtype=dtype_out, device="cuda")
    bias = Tensor(shape=(N,), dtype="f32", device="cuda") if has_bias else None

    g = Graph()
    g.add_op(
        "GEMM_BIAS_ACT",
        inputs=[A, B] + ([bias] if bias else []),
        outputs=[C],
        attrs={"mnk": (M, N, K), "act": act},
    )
    return g, A, B, bias, C
