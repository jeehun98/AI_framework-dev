# File: python/graph_executor_v2/layers/conditional.py
from __future__ import annotations
from typing import Callable, Tuple, Optional, Iterable, Any, Dict, List
from .base import Layer

class _MetaControlLayer(Layer):
    """
    공통 유틸: control 레이어들은 실제 연산을 수행하지 않고
    '경로 평탄화(Sequential._linearize_path)' 단계에서만 사용됩니다.

    - __call__/forward는 사용 시도 자체를 막기 위해 예외를 던집니다.
    - parameters/zero_grad/train은 내부 블록으로 위임될 수 있도록 각 서브클래스에서 override합니다.
    """
    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"{self.__class__.__name__} is a control-only layer. "
            "It should not be executed directly. Use dynamic path handling "
            "(Sequential.one_step_dynamic) which linearizes the path first."
        )

    def forward(self, *args, **kwargs):
        # base.Layer가 forward를 호출하는 경우를 방지
        return self.__call__(*args, **kwargs)

    def parameters(self) -> Iterable[Tuple[Any, Any, str]]:
        # 내부 블록/스테이지의 파라미터를 합쳐 반환하도록 각 서브클래스에서 override
        return tuple()

    def zero_grad(self):
        # 내부 블록/스테이지의 zero_grad를 호출하도록 각 서브클래스에서 override
        pass

    def train(self, mode: bool = True):
        # 내부 블록/스테이지로 전파하도록 각 서브클래스에서 override
        super().train(mode)
        return self


class If(_MetaControlLayer):
    """
    Python 분기 레이어 (캡처 밖에서 분기 결정).
    cond_fn(X, ctx) -> bool
    then_block / else_block: Layer 또는 Sequential

    요구사항:
      - then/else 블록의 출력 shape은 동일해야 합니다(정합성 강제).
    """
    def __init__(self, cond_fn: Callable[[Any, Dict[str, Any]], bool],
                 then_block: Layer, else_block: Layer, name: Optional[str] = None):
        super().__init__(name=name or "If")
        self.cond_fn = cond_fn
        self.then_block = then_block
        self.else_block = else_block

    # 캡처 밖: 분기 결정
    def decide(self, X, ctx) -> Tuple[str, Layer]:
        cond = bool(self.cond_fn(X, ctx))
        if cond:
            ctx["branch"] = "then"
            return "then", self.then_block
        else:
            ctx["branch"] = "else"
            return "else", self.else_block

    # build/shape: 두 경로가 동일 출력 shape인지 확인
    def build(self, input_shape: Tuple[int, ...], **kwargs) -> None:
        ish = tuple(map(int, input_shape))
        if hasattr(self.then_block, "build"):
            self.then_block.build(ish, **kwargs)  # type: ignore
        if hasattr(self.else_block, "build"):
            self.else_block.build(ish, **kwargs)  # type: ignore

        # shape 추론 검증
        try:
            osh_then = tuple(map(int, self.then_block.compute_output_shape(ish)))
        except Exception as e:
            raise RuntimeError(f"If.then_block.compute_output_shape failed: {e}") from e
        try:
            osh_else = tuple(map(int, self.else_block.compute_output_shape(ish)))
        except Exception as e:
            raise RuntimeError(f"If.else_block.compute_output_shape failed: {e}") from e

        if osh_then != osh_else:
            raise RuntimeError(
                f"If branches must produce the same output shape, "
                f"got then={osh_then}, else={osh_else}"
            )

        self.input_shape = ish
        self.output_shape = osh_then
        self.built = True

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        # build에서 검증했으므로 어느 한쪽을 반환
        ish = tuple(map(int, input_shape))
        return tuple(map(int, self.then_block.compute_output_shape(ish)))

    # 경로 예열용: 입출력 관찰/등록 (선형화 시 활용)
    def peek_io(self, ctx):
        return None

    # 모드/파라미터 전파
    def train(self, mode: bool = True):
        super().train(mode)
        for blk in (self.then_block, self.else_block):
            if hasattr(blk, "training"):
                blk.training = bool(mode)
            if hasattr(blk, "train"):
                blk.train(mode)  # type: ignore
        return self

    def parameters(self) -> Iterable[Tuple[Any, Any, str]]:
        # 두 블록의 파라미터를 합쳐서 방출
        for idx, blk in enumerate((self.then_block, self.else_block)):
            lname = f"{self.name}.branch{idx}"
            if hasattr(blk, "parameters"):
                for t in blk.parameters():  # type: ignore
                    if isinstance(t, tuple) and len(t) == 3:
                        yield t
                    elif isinstance(t, tuple) and len(t) == 2:
                        p, g = t
                        yield (p, g, lname)

    def zero_grad(self):
        for blk in (self.then_block, self.else_block):
            if hasattr(blk, "zero_grad"):
                blk.zero_grad()  # type: ignore


class Repeat(_MetaControlLayer):
    """
    반복 레이어. 본문(body)의 '1 step'을 캡처하고 Python에서 T회 launch.
    steps_fn(X, ctx) -> int (반복 횟수)
    """
    def __init__(self, body: Layer, steps_fn: Callable[[Any, Dict[str, Any]], int],
                 name: Optional[str] = None):
        super().__init__(name=name or "Repeat")
        self.body = body
        self.steps_fn = steps_fn

    def steps(self, X, ctx) -> int:
        T = int(self.steps_fn(X, ctx))
        if T < 1:
            # 최소 1회는 실행(부분 캡처/런칭 의미 유지)
            return 1
        return T

    def build(self, input_shape: Tuple[int, ...], **kwargs) -> None:
        ish = tuple(map(int, input_shape))
        if hasattr(self.body, "build"):
            self.body.build(ish, **kwargs)  # type: ignore
        self.input_shape = ish
        self.output_shape = tuple(map(int, self.body.compute_output_shape(ish)))
        self.built = True

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        ish = tuple(map(int, input_shape))
        return tuple(map(int, self.body.compute_output_shape(ish)))

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self.body, "training"):
            self.body.training = bool(mode)
        if hasattr(self.body, "train"):
            self.body.train(mode)  # type: ignore
        return self

    def parameters(self) -> Iterable[Tuple[Any, Any, str]]:
        if hasattr(self.body, "parameters"):
            for t in self.body.parameters():  # type: ignore
                yield t

    def zero_grad(self):
        if hasattr(self.body, "zero_grad"):
            self.body.zero_grad()  # type: ignore


class EarlyExit(_MetaControlLayer):
    """
    다단(stage) 레이어. 각 stage를 순차 실행하다가 Python에서 exit_fn(ctx)==True면 조기 종료.
    - 최소 구현: shape 추론은 모든 stage를 통과한다고 가정(마지막 stage 출력).
    - 고급 구현: stage별 부분 캡처/실행은 Sequential._linearize_path와 그래프 풀에서 처리.
    """
    def __init__(self, stages: List[Layer], exit_fn: Callable[[Dict[str, Any]], bool],
                 name: Optional[str] = None):
        super().__init__(name=name or "EarlyExit")
        assert isinstance(stages, list) and len(stages) > 0, "stages must be non-empty list"
        self.stages = stages
        self.exit_fn = exit_fn

    def build(self, input_shape: Tuple[int, ...], **kwargs) -> None:
        cur = tuple(map(int, input_shape))
        for i, s in enumerate(self.stages):
            if hasattr(s, "build"):
                s.build(cur, **kwargs)  # type: ignore
            cur = tuple(map(int, s.compute_output_shape(cur)))
        self.input_shape = tuple(map(int, input_shape))
        self.output_shape = cur
        self.built = True

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        cur = tuple(map(int, input_shape))
        for s in self.stages:
            cur = tuple(map(int, s.compute_output_shape(cur)))
        return cur

    def train(self, mode: bool = True):
        super().train(mode)
        for s in self.stages:
            if hasattr(s, "training"):
                s.training = bool(mode)
            if hasattr(s, "train"):
                s.train(mode)  # type: ignore
        return self

    def parameters(self) -> Iterable[Tuple[Any, Any, str]]:
        for i, s in enumerate(self.stages):
            lname = f"{self.name}.stage{i}"
            if hasattr(s, "parameters"):
                for t in s.parameters():  # type: ignore
                    if isinstance(t, tuple) and len(t) == 3:
                        yield t
                    elif isinstance(t, tuple) and len(t) == 2:
                        p, g = t
                        yield (p, g, lname)

    def zero_grad(self):
        for s in self.stages:
            if hasattr(s, "zero_grad"):
                s.zero_grad()  # type: ignore
