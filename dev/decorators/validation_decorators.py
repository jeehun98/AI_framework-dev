import numpy as np

def data_validation(func):
    def wrapper(data, *args, **kwargs):
        # 데이터가 numpy 배열인지 확인
        if not isinstance(data, np.ndarray):
            raise ValueError(f"Input data must be a numpy array, got {type(data)} instead.")
        
        # 데이터가 비어 있지 않은지 확인
        if data.size == 0:
            raise ValueError("Input data must not be empty.")
        
        # 추가적인 데이터 검증이 필요한 경우 여기에 추가 가능
        # 예: 데이터가 양수만 포함해야 하는 경우
        # if np.any(data < 0):
        #     raise ValueError("Input data must contain only non-negative values.")
        
        # 원래 함수 호출
        return func(data, *args, **kwargs)
    return wrapper

def check_layer_type(func):
    def wrapper(self, layer, *args, **kwargs):
        if not isinstance(layer, Layer):
            raise TypeError(f"Expected an instance of Layer, got {type(layer).__name__}")
        # 원하는 추가 검사를 여기에 추가할 수 있습니다.
        print(f"Adding layer: {layer.name}")
        return func(self, layer, *args, **kwargs)
    return wrapper