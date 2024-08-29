import inspect

from dev.losses.losses import CategoricalCrossentropy

ALL_LOSSES= {
    CategoricalCrossentropy,
}

ALL_LOSSES_DICT = {cls.__name__.lower(): cls for cls in ALL_LOSSES}

def get(identifier):
    if isinstance(identifier, str):
        obj = ALL_LOSSES_DICT.get(identifier, None)

    if callable(obj):
        # 클래스일 경우 클래스의 인스턴스화, 
        if inspect.isclass(obj):
            obj = obj()
        return obj
    else:
        raise ValueError(f"Could not interpret loss identifier: {identifier}")