from dev import regularizers

class Layer():
    
    def __new__(cls, *args, **kwargs):
        
        # build_wrapper 의 구현??
        pass
    
    def __init__(self, name=None, regularizer=None,**kwargs):
        self.name = name
        self.input_shape = None
        self.output_shape = None
        self.regularizer = regularizers.get(regularizer)
        input_dim_arg = kwargs.pop("input_dim", None)
        if input_dim_arg is not None:
            input_dim_arg = (input_dim_arg,)


    # 가중치 초기화
    def build(self):
        # 이미 저장되어 있는 인스턴스로부터 input_shape, output_shape 를 계산할 수 있어야
        
        pass
    
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