from dev import optimizers
from dev.trainer.data_adapters.data_adapter_util import unpack_x_y_sample_weight

class Trainer:
    def __init__(self):
        self.loss = None
        self.compiled = False
        
    # 모델 훈련을 위해 필요한 옵티마이저, 손실 함수, mterics 의 설정
    def compile(self, optimizer="rmsprop", loss=None, metrics = None):
        self.optimzier = optimizers.get(optimizer)

        self.compiled = True

    # 한 배치의 학습의 수행
    def train_step(self, data):
        x, y, sample_weight = unpack_x_y_sample_weight(data)
        

    def fit(self, x=None, y=None):
        pass