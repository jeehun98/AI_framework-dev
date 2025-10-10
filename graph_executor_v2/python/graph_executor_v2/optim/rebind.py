# python/graph_executor_v2/optim/rebind.py
from __future__ import annotations
from typing import Any, List, Tuple

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
