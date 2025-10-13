# python/graph_executor_v2/optim/rebind.py
from __future__ import annotations
from typing import Any, List, Tuple

# 옵티마이저 비종속으로, 캡처 플랜의 gW/gB 버퍼를 옵티마이저가 읽는 grad 포인터로 재바인딩하는 유틸.
# 다른 옵티마이저(SGD 등)도 rebind_grads만 구현하면 재바인딩 경로에 그대로 편입됨.
# 레이어가 W/b 외 파라미터를 갖는다면, 해당 레이어 타입에 맞춘 매칭 로직을 추가.

try:
    # 옵티마이저 구현에서 제공하는 plan 기반 수집 유틸(있으면 우선 사용)
    from .adamw import collect_params_from_plan as _collect_from_adamw  # type: ignore
except Exception:
    _collect_from_adamw = None

def collect_params_from_plan(model, plan) -> List[Tuple[Any, Any, bool]]:
    """
    캡처 플랜에서 (param, grad, exempt) triplet 추출.
    - AdamW가 별도 수집 유틸을 제공하면 그걸 사용
    - 아니면 Dense 휴리스틱(W/b 매칭)으로 수집
    """
    if _collect_from_adamw is not None:
        try:
            return _collect_from_adamw(model, plan)  # type: ignore
        except Exception:
            pass

    triplets: List[Tuple[Any, Any, bool]] = []
    for lyr, per in zip(model.layers, plan.per_layer):
        if hasattr(lyr, "W") and per.gW is not None:
            triplets.append((lyr.W, per.gW, False))
        if hasattr(lyr, "b") and per.gB is not None:
            triplets.append((lyr.b, per.gB, True))
    return triplets

def try_rebind_grads(model, optimizer, plan) -> None:
    """옵티마이저가 rebind_grads를 지원하면 plan 기반 gW/gB 포인터로 재바인딩."""
    if hasattr(optimizer, "rebind_grads"):
        triplets = collect_params_from_plan(model, plan)
        optimizer.rebind_grads(triplets)
