class Layer():
    def __init__(self, name, input_shape =None):
        self.name = name
        self.input_shape = input_shape
        self.weights = None
        self.trainable = True

    # 가중치 초기화
    def build(self):
        pass
    
    # 연산
    def call(self):
        pass

    def get_config(self):
        # 레이어의 구성 정보 반환
        return {
            "name": self.name,
            "input_shape": self.input_shape,
            "trainable": self.trainable
        }

    @classmethod
    def from_config(cls, config):
        # 구성 정보로부터 레이어 생성
        return cls(**config)