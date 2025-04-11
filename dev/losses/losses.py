# dev/losses/losses.py

import os
import sys
import numpy as np

# ✅ test_setup.py 경로 추가
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev/dev/tests"))

from test_setup import import_cuda_module

# ✅ losses_cuda 모듈 import
losses_cuda = import_cuda_module(
    module_name="losses_cuda",
    build_dir="C:/Users/owner/Desktop/AI_framework-dev/dev/backend/backend_ops/losses/build/lib.win-amd64-cpython-312"
)


class MSE:
    def __init__(self, name="mse"):
        self.name = name

    def get_config(self):
        return {"name": self.name}

    def __call__(self, y_true, y_pred):
        return losses_cuda.compute_loss(y_true.astype(np.float32), y_pred.astype(np.float32), "mse")


class BinaryCrossentropy:
    def __init__(self, name="binarycrossentropy"):
        self.name = name

    def get_config(self):
        return {"name": self.name}

    def __call__(self, y_true, y_pred):
        return losses_cuda.compute_loss(y_true.astype(np.float32), y_pred.astype(np.float32), "bce")


class CategoricalCrossentropy:
    def __init__(self, name="categoricalcrossentropy"):
        self.name = name

    def get_config(self):
        return {"name": self.name}

    def __call__(self, y_true, y_pred):
        return losses_cuda.compute_loss(y_true.astype(np.float32), y_pred.astype(np.float32), "cce")
