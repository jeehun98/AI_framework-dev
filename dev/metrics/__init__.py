import inspect

from dev.metrics.accuracy_metrics import Accuracy
from dev.metrics.accuracy_metrics import MSE

ALL_METRICS = {
    Accuracy,
    MSE,
}

ALL_METRICS_DICT = {cls.__name__.lower(): cls for cls in ALL_METRICS}

def get(identifier):
    if isinstance(identifier, str):
        obj = ALL_METRICS_DICT.get(identifier, None)
    
    if callable(obj):
        if inspect.isclass(obj):
            obj = obj()
        return obj
    else:
        raise ValueError(f"Could not interpret metric identifier: {identifier}")