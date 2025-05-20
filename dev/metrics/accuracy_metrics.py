# backend 변환
import os
os.add_dll_directory("C:\\msys64\\mingw64\\bin")

from dev.backend.backend_ops.metrics import metrics

class Accuracy():
    def __init__(self, name="accuracy"):
        self.name = name

    def get_config(self):
        return {"name" : self.name}
    
class MSE():
    def __init__(self, name="mse"):
        self.name = name

    def get_config(self):
        return {"name" : self.name}
    
    def __call__(self, y_true, y_pred):
        """
        MSE 클래스를 호출할 때 C++의 mean_squared_error 함수를 호출하도록 구성합니다.
        """
        return metrics.mse(y_true, y_pred)