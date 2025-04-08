import os
os.add_dll_directory("C:\\msys64\\mingw64\\bin")

import inspect

from dev.backend.backend_ops.optimizers.optimizers import SGD

ALL_OPTIMIZERS = {
    SGD,
}

ALL_OPTIMIZERS_DICT = {cls.__name__.lower(): cls for cls in ALL_OPTIMIZERS}

def get(identifier, **kwargs):
    if isinstance(identifier, str):
        obj = ALL_OPTIMIZERS_DICT.get(identifier.lower(), None)
        
    if callable(obj):
        if inspect.isclass(obj):
            obj = obj(**kwargs)  # 인스턴스를 생성할 때 추가 인자를 전달
        return obj
    else:
        raise ValueError(f"Could not interpret optimizer identifier: {identifier}")