from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence, Any, Dict, Tuple

StepKind = Literal["fwd", "loss", "bwd", "opt", "barrier"]  # ← 확장 여지 확보

@dataclass(frozen=True)
class ExecStep:
    """
    한 스텝의 원자적 실행 단위.
      - kind: "fwd" | "loss" | "bwd" | "opt" | "barrier"(옵션)
      - layer_idx: 레이어 인덱스 (loss/opt/barrier는 None)
      - stream_id: 실행할 CUDA 스트림 ID (0..num_streams-1)
    """
    kind: StepKind
    layer_idx: Optional[int]
    stream_id: int = 0

@dataclass(frozen=True)
class ExecPlan:
    """
    실행계획 컨테이너.
      - num_streams: 사용할 CUDA 스트림 수
      - steps: ExecStep의 순서 리스트(런타임이 그대로 순회)
      - metadata: 디버그/관찰용 부가정보(선택)
    """
    num_streams: int
    steps: Tuple[ExecStep, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_streams": self.num_streams,
            "steps": [step.__dict__ for step in self.steps],
            "metadata": dict(self.metadata),
        }

class ExecPlanner:
    """
    축소 버전 Execution Planner:
      - 입력 CapturePlan을 참고하여 선형 순서의 ExecPlan을 만든다.
      - 현재는 다중 스트림/의존성 분석은 하지 않고 stream_id=0 고정.
      - 확장 포인트: 독립 레이어 분석 → 다중 스트림/이벤트/프리페치 등.
    """

    def __init__(self) -> None:
        pass

    def build(
        self,
        *,
        plan: Any,
        max_streams: int = 1,
        layers_override: Optional[Sequence[Any]] = None,
        loss_after: Literal["fwd", "bwd"] = "fwd",   # ← loss 위치 선택
        include_loss: bool = True,                  # ← 추론시 False
        include_opt: bool = True,                   # ← 평가/추론시 False
        metadata: Optional[Dict[str, Any]] = None,  # ← 디버그/로깅
    ) -> ExecPlan:
        """
        선형 스케줄 생성:
          기본:  fwd(L0..Lk) → (loss) → bwd(Lk..L0) → (opt)
          옵션:  loss 위치(loss_after) 조정 및 단계 생략
        """
        # --- 입력 검증 ---
        try:
            per_layer = getattr(plan, "per_layer")
        except Exception as e:
            raise ValueError("plan must have 'per_layer' attribute") from e

        num_layers = len(per_layer)
        if layers_override is not None:
            if len(layers_override) != num_layers:
                raise ValueError(
                    f"layers_override length mismatch: "
                    f"{len(layers_override)} vs plan.per_layer={num_layers}"
                )

        n_streams = int(max(1, int(max_streams)))
        steps: List[ExecStep] = []

        # --- Forward ---
        for i in range(num_layers):
            steps.append(ExecStep(kind="fwd", layer_idx=i, stream_id=0))

        # --- Loss (옵션, 위치 선택) ---
        if include_loss and loss_after == "fwd":
            steps.append(ExecStep(kind="loss", layer_idx=None, stream_id=0))

        # --- Backward ---
        # (loss_after == "bwd"인 경우, 일부 커스텀 그래프에서 post-fwd-reduce 후 계산 가능성 대비)
        for i in reversed(range(num_layers)):
            steps.append(ExecStep(kind="bwd", layer_idx=i, stream_id=0))

        if include_loss and loss_after == "bwd":
            steps.append(ExecStep(kind="loss", layer_idx=None, stream_id=0))

        # --- Optimizer (옵션) ---
        if include_opt:
            steps.append(ExecStep(kind="opt", layer_idx=None, stream_id=0))

        # metadata 안전 채움
        md = dict(metadata or {})
        # 유용 메타 키 예시: path_fingerprint, variant, tag 등
        if "num_layers" not in md:
            md["num_layers"] = num_layers

        return ExecPlan(num_streams=n_streams, steps=tuple(steps), metadata=md)

__all__ = ["ExecPlanner", "ExecPlan", "ExecStep", "StepKind"]
