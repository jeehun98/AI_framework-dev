import inspect

from dev.regularizers.regularizers import L1
from dev.regularizers.regularizers import L2
from dev.regularizers.regularizers import L1L2
from dev.regularizers.regularizers import Orthogonal
from dev.regularizers.regularizers import Regularizer


ALL_REGULARIZERS = {
    L1,
    L2,
    L1L2,
    Orthogonal,
    Regularizer,
}

ALL_REGULARIZERS_DICT = {cls.__name__: cls for cls in ALL_REGULARIZERS}


def get(identifier):
    if isinstance(identifier, str):
        # 예를 들어, identifier가 문자열인 경우, 이에 해당하는 regularizer 객체를 가져옴
        obj = ALL_REGULARIZERS_DICT.get(identifier, None)
    else:
        # identifier가 문자열이 아니면, identifier 자체가 regularizer 객체일 가능성
        obj = identifier
    
    if callable(obj):
        return obj
    else:
        raise ValueError(f"Could not interpret regularizer identifier: {identifier}")
