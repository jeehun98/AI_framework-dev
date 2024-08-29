class SGD():
    def __init__(
        self,
        learning_rate = 0.01,
        momentum = 0.0,
        name = "SGD",
        **kwargs,
    ):
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.name = name

    def get_config(self):
        config = {
            "learning_rate" : self.learning_rate,
            "momentum" : self.momentum,
            "name" : self.name
        } 
        return config

    # optimizer 변수 초기화
    def build(self, variables):
        pass

    # 주어진 그래디언트와 모델 변수에 대해 업데이트 수행
    def update_step(self, gradient, variable, learning_rate):
        pass

