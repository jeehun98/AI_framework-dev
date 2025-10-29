from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Any

# 이 모듈은 "스케줄을 만든다"에만 집중합니다.
# - 그래프 토폴로지 분석 없이, 단일 스트림 선형 스케줄을 생성
# - 향후 다중 스트림/이벤트/프리페치 등을 확장할 발판

StepKind = Literal["fwd", "loss", "bwd", "opt"]


@dataclass
class ExecStep:
    """
    한 스텝의 원자적 실행 단위.
      - kind: "fwd" | "loss" | "bwd" | "opt"
      - layer_idx: 레이어 인덱스 (loss/opt는 None)
      - stream_id: 실행할 CUDA 스트림 ID (0..num_streams-1)
    """
    kind: StepKind
    layer_idx: Optional[int]
    stream_id: int = 0


@dataclass
class ExecPlan:
    """
    실행계획 컨테이너.
      - num_streams: 사용할 CUDA 스트림 수 (축소판: 항상 1을 권장)
      - steps: ExecStep의 순서 리스트(런타임이 그대로 순회)
    """
    num_streams: int
    steps: List[ExecStep]


class ExecPlanner:
    """
    축소 버전 Execution Planner:
      - 입력 CapturePlan을 참고하여 선형 순서의 ExecPlan을 만든다.
      - 현재는 다중 스트림/의존성 분석은 하지 않고 stream_id=0 고정.
      - 향후 확장: 독립 레이어 분석 → 다중 스트림, 이벤트 동기화, 프리페치 등.
    """

    def __init__(self):
        pass

    def build(
        self,
        *,
        plan: Any,
        max_streams: int = 1,
        layers_override: Optional[Sequence[Any]] = None,
    ) -> ExecPlan:
        """
        선형 스케줄 생성:
          fwd(L0) ... fwd(Lk) → loss → bwd(Lk) ... bwd(L0) → opt
        """
        # 스트림 수는 축소판에서는 1 권장(추후 확장 포인트)
        n_streams = max(1, int(max_streams))
        steps: List[ExecStep] = []

        # 레이어 수는 plan.per_layer 길이에 의존 (동적 경로: layers_override와 길이 동일해야 함)
        num_layers = len(getattr(plan, "per_layer", []))

        # Forward
        for i in range(num_layers):
            steps.append(ExecStep(kind="fwd", layer_idx=i, stream_id=0))

        # Loss
        steps.append(ExecStep(kind="loss", layer_idx=None, stream_id=0))

        # Backward (역순)
        for i in reversed(range(num_layers)):
            steps.append(ExecStep(kind="bwd", layer_idx=i, stream_id=0))

        # Optimizer
        steps.append(ExecStep(kind="opt", layer_idx=None, stream_id=0))

        return ExecPlan(num_streams=n_streams, steps=steps)


__all__ = ["ExecPlanner", "ExecPlan", "ExecStep", "StepKind"]
