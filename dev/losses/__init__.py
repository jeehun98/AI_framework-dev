# dev/losses/__init__.py

import inspect

from dev.losses.losses import CategoricalCrossentropy
from dev.losses.losses import BinaryCrossentropy
from dev.losses.losses import MSE

# ✅ 등록된 손실 클래스 목록
ALL_LOSSES = [
    CategoricalCrossentropy,
    BinaryCrossentropy,
    MSE,
]

# ✅ 이름 → 클래스 매핑 딕셔너리
ALL_LOSSES_DICT = {cls.__name__.lower(): cls for cls in ALL_LOSSES}

def get(identifier):
    """
    문자열 또는 클래스/함수 객체를 받아 손실 함수 인스턴스를 반환합니다.

    Parameters:
        identifier (str or callable): 손실 함수 이름 또는 클래스/함수 객체

    Returns:
        callable: 손실 함수 인스턴스 또는 함수
    """
    if isinstance(identifier, str):
        identifier = identifier.lower()
        obj = ALL_LOSSES_DICT.get(identifier)

        if obj is None:
            raise ValueError(
                f"Invalid loss identifier: '{identifier}'. "
                f"Available options: {', '.join(ALL_LOSSES_DICT.keys())}"
            )

        if inspect.isclass(obj):
            return obj()

    elif callable(identifier):
        return identifier

    raise ValueError(f"Could not interpret loss identifier: {identifier}")
