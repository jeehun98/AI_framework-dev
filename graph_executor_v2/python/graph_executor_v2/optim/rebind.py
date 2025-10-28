# File: python/graph_executor_v2/optim/rebind.py
from __future__ import annotations
from typing import Any, List, Tuple, Optional, Callable

__all__ = [
    "collect_params_from_plan",
    "try_rebind_grads",
]

# -----------------------------------------------------------------------------
# 이 파일의 역할
# -----------------------------------------------------------------------------
# - 캡처 플랜(CapturePlan)의 gW/gB(그리고 일부 레이어의 특수 grad 버퍼)를
#   옵티마이저가 직접 읽을 수 있는 grad 포인터로 "재바인딩(rebind)" 해 준다.
# - 옵티마이저가 rebind_grads(triplets) API를 제공하면 여기서 만든 triplets를 전달.
#   triplet 스키마: (param, grad_buffer, exempt_from_weight_decay: bool)
# - AdamW 전용의 수집기가 있으면 우선 사용하고, 없으면 휴리스틱 기반 수집을 사용한다.
#
# 참고
# - 동적 경로(one_step_dynamic)에서는 경로별 정확 매칭이 중요해, sequential.py에서
#   별도의 수집 로직을 이미 사용하고 있음(여기 로직을 그대로 쓰지 않는다).
# -----------------------------------------------------------------------------

# (선택) 옵티마이저가 제공하는 전용 수집기: 있으면 우선 사용
try:
    # adamw.py 쪽에서 plan을 해석해 (param, grad, exempt) triplets를 만들어 주는 헬퍼
    from .adamw import collect_params_from_plan as _collect_from_adamw  # type: ignore
except Exception:
    _collect_from_adamw = None


# ---- 유틸: 안전한 속성 획득 -------------------------------------------------
def _get(obj: Any, name: str) -> Optional[Any]:
    return getattr(obj, name, None) if hasattr(obj, name) else None


def _has_tensor(x: Any) -> bool:
    """param/grad 후보가 None이 아니고 크기/버퍼를 가진 것으로 가정 가능한지 대략 판정."""
    return x is not None and hasattr(x, "shape")


# ---- 휴리스틱 기반 수집기 ----------------------------------------------------
def collect_params_from_plan(model, plan) -> List[Tuple[Any, Any, bool]]:
    """
    CapturePlan(per_layer)로부터 (param, grad, exempt) triplets를 수집한다.

    우선순위:
      1) 옵티마이저(AdamW 등)가 제공하는 전용 수집기가 있으면 그걸 사용.
      2) 아니면 아래 휴리스틱으로 Dense/Conv2D/BN/Embedding/RNN 등을 커버.

    exempt 의미:
      - weight decay에서 제외할지 힌트(True면 보통 WD 제외).
      - 일반적으로 bias, BN(beta/gamma)은 WD 제외가 기본값이므로 True로 분류.
        (프로젝트 정책에 따라 달리면 옵티마이저 쪽에서 무시/재해석해도 됨)
    """
    if _collect_from_adamw is not None:
        try:
            return _collect_from_adamw(model, plan)  # type: ignore
        except Exception:
            # 전용 수집기가 실패하면 휴리스틱 경로로 폴백
            pass

    triplets: List[Tuple[Any, Any, bool]] = []

    # 이름 패턴 모음: 다양한 레이어 구현의 편차를 흡수
    WEIGHT_NAMES = ("W", "weight")
    BIAS_NAMES   = ("b", "bias", "B")
    # BN, LayerNorm 등
    GAMMA_NAMES  = ("gamma", "weight")   # 일부 구현은 weight가 scale(=gamma)
    BETA_NAMES   = ("beta", "bias")      # 일부 구현은 bias가 beta

    for lyr, per in zip(getattr(model, "layers", []), plan.per_layer):
        # ---- 1) Dense/Conv 계열: W / (b|bias|B)
        if per.gW is not None:
            pW = None
            for nm in WEIGHT_NAMES:
                val = _get(lyr, nm)
                if _has_tensor(val):
                    pW = val
                    break
            if _has_tensor(pW):
                # 가중치는 보통 WD 적용 → exempt=False
                triplets.append((pW, per.gW, False))

        if per.gB is not None:
            pb = None
            for nm in BIAS_NAMES:
                val = _get(lyr, nm)
                if _has_tensor(val):
                    pb = val
                    break
            if _has_tensor(pb):
                # bias는 보통 WD 제외 → exempt=True
                triplets.append((pb, per.gB, True))

        # ---- 2) BatchNorm / LayerNorm 계열: gamma/beta
        # per.gW/gB가 BN 용도로 재사용되는 플랜 구현을 지원
        # (capture_plan에서 BN은 gW=dgamma, gB=dbeta 로 잡혀 있음)
        if per.gW is not None:
            pgamma = None
            for nm in GAMMA_NAMES:
                val = _get(lyr, nm)
                if _has_tensor(val) and val.shape == per.gW.shape:
                    pgamma = val
                    break
            if _has_tensor(pgamma):
                # BN/LN scale(gamma)은 WD 제외가 일반적 → exempt=True
                triplets.append((pgamma, per.gW, True))

        if per.gB is not None:
            pbeta = None
            for nm in BETA_NAMES:
                val = _get(lyr, nm)
                if _has_tensor(val) and val.shape == per.gB.shape:
                    pbeta = val
                    break
            if _has_tensor(pbeta):
                # BN/LN bias(beta)도 WD 제외 → exempt=True
                triplets.append((pbeta, per.gB, True))

        # ---- 3) RNN(Elman) 계열: Wx/Wh, b, (옵션) 초기 은닉 dh0
        # capture_plan에서 gWx/gWh/dh0가 존재할 수 있음
        if _has_tensor(getattr(per, "gWx", None)) and _has_tensor(_get(lyr, "Wx")):
            triplets.append((lyr.Wx, per.gWx, False))
        if _has_tensor(getattr(per, "gWh", None)) and _has_tensor(_get(lyr, "Wh")):
            triplets.append((lyr.Wh, per.gWh, False))
        if _has_tensor(getattr(per, "gB", None)) and _has_tensor(_get(lyr, "b")):
            # RNN bias → WD 제외
            triplets.append((lyr.b, per.gB, True))
        # dh0 (학습하려면 param으로 가진 구현일 때만)
        if _has_tensor(getattr(per, "dh0", None)) and _has_tensor(_get(lyr, "h0")):
            # 초기 은닉 상태를 파라미터로 두는 구현이라면 WD 제외 권장
            triplets.append((lyr.h0, per.dh0, True))

        # ---- 4) Embedding: W(=table), grad는 plan.gW (per.gW에 담기지 않을 수 있음)
        # capture_plan의 Embedding 플래너는 gW=(V,D)로 설정됨.
        if per.gW is not None:
            # ndarray에 대해 'a or b'를 쓰면 truth value 평가로 ValueError 발생 → 명시 분기
            pE = _get(lyr, "W")
            if _has_tensor(pE):
                pE = _get(lyr, "weight")
            if _has_tensor(pE) and pE.shape == per.gW.shape:
                # WD 적용 여부는 정책에 따라. 여기서는 기본 False(적용)로 두고 옵티마이저가 exempt 해석 가능.
                triplets.append((pE, per.gW, False))            
            

    return triplets


def try_rebind_grads(model, optimizer, plan) -> None:
    """
    옵티마이저가 rebind_grads(triplets)를 지원하면,
    plan 기반 grad 버퍼들로 파라미터의 grad 포인터를 재바인딩한다.

    - rebind_grads가 없으면 NO-OP.
    - 옵티마이저에 ensure_initialized()가 있다면, 재바인딩 전에 한 번 호출 시도(옵션).
    """
    if not hasattr(optimizer, "rebind_grads"):
        return

    # (옵션) 일부 옵티마이저는 내부 버퍼 초기화가 필요할 수 있음
    if hasattr(optimizer, "ensure_initialized"):
        try:
            optimizer.ensure_initialized()
        except Exception:
            pass

    triplets = collect_params_from_plan(model, plan)
    optimizer.rebind_grads(triplets)
