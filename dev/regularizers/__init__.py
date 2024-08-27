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
    if identifier is isinstance(str):
        obj = ALL_REGULARIZERS_DICT.get(identifier, None)

    if callable(obj):
        if inspect.isclass(obj):
            obj = obj()
        return obj

    else:
        raise ValueError(
            f"Could not interpret regularizer identifier: {identifier}"
        )