# shim so that `from graph_executor_v2.train.capture_trainer import ...` works
from .graph_capture_trainer import *  # re-export everything
