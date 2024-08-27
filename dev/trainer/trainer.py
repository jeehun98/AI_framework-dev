from dev import optimizers

class Trainer:
    def __init__(self):
        self.loss = None
        self.compiled = False
        
    # 모델 훈련을 위해 필요한 옵티마이저, 손실 함수, mterics 의 설정
    def compile(self, optimizer="rmsprop", loss=None, metrics = None):
        self.optimzier = optimizers.get(optimizer)

        self.compiled = True