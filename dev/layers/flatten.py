from dev.layers.layer import Layer
import numpy as np

class Flatten(Layer):
    def __init__(self, input_shape=None, **kwargs):
        # 인스턴스 생성 시 어떤 값이 추가되어야 할 지에 대한 고민
        super().__init__(input_shape, **kwargs)
        self.input_shape = input_shape

    # 객체 정보를 저장하는 get_config
    # 향후 from_config 를 통해 해당 객체를 복원할 수 있다. 
    def get_config(self):
        base_config = super().get_config()
        config = {
            "class_name": self.__class__.__name__,
            "input_shape": self.input_shape
        }
        return {**base_config, **config}

    # 저장된 config 파일로 객체 생성
    def from_config(cls, config):
        return(cls **config)

    # 계산시 (batch_size, flattened_dim) 형태,
    # 전체 데이터에 대해 생각해야해, 데이터 하나에 대한 flatten 연산이 아님

    # 여러 데이터의 입력으로 확장시킬 수 있을 것
    def call(self, inputs):
        """
        p~ 차원 입력 데이터를 펼침
        
        Parameters:
        inputs (n., p) 
               (n, p_1, p_2) : 특정 차원별 n개의 데이터가 입력으로 들어옴 - 배치!!
        
        Returns:
        (n, p*) : 펼친 데이터, 1개의 데이터가 입력 될 경우 행 벡터 출력
        """
        shape = inputs.shape

        n = shape[0]

        flatten_size = np.prod(shape[1:])

        # 가중치와의 연산을 위한 적절하 변환
        flattened_array = inputs.reshape(n, -1, flatten_size)

        return flattened_array

    def compute_output_shape(self, input_shape):
        # 입력 shape 를 기반으로 출력 shape 를 계산, 모델의 구조 정의
        return (input_shape[0], np.prod(input_shape[1:]))

    
    # build 는 가중치 초기화의 정의, 
    # flatten 은 별도의 가중치를 필요로 하지 않음
    def build(self):
        pass