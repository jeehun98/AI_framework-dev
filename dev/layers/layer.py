class Layer():
    def __init__(self, name=None):
        self.name = name
        self.input_shape = None
        self.output_shape = None

    # 가중치 초기화
    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = self.compute_output_shape(input_shape)

    
    # 연산
    def call(self):
        pass

    def get_config(self):
        # 레이어의 구성 정보 반환
        return {
            'name': self.name,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
        }

    @classmethod
    def from_config(cls, config):
        instance = cls(name=config['name'])
        instance.input_shape = config['input_shape']
        instance.output_shape = config['output_shape']
        return instance