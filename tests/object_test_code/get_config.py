class MyLayer:
    def __init__(self, units, activation):
        self.units = units
        self.activation = activation

    def get_config(self):
        return {
            'units': self.units,
            'activation': self.activation
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# 객체 생성
layer = MyLayer(units=64, activation='relu')

# 구성 정보 가져오기
config = layer.get_config()
print(config)

# 구성 정보를 사용하여 객체 복원
new_layer = MyLayer.from_config(config)
print(new_layer.units, new_layer.activation)
