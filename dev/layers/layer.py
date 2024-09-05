from dev import regularizers
import collections

class Layer():
    
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        
        # 추가적인 동작 수행

        return obj
    
    
    def __init__(self, name=None, regularizer=None,**kwargs):
        self.name = name
        self.input_shape = None
        self.output_shape = None
        if regularizer is not None:
            self.regularizer = regularizers.get(regularizer)
        else:
            self.regularizer = None
        input_dim_arg = kwargs.pop("input_dim", None)
        if input_dim_arg is not None:
            input_dim_arg = (input_dim_arg,)

    # layer build 는 뭘 추가해야 할지
    # layer 가 build 되었는지에 대해..
    def build(self):
       self.built = True

    
    # 연산이 실행되는 부분, layer 를 상속받는 클래스에서 이를 구현해야 한다. 
    def call(self, *args, **kwargs):
        raise NotImplementedError(
            f"Layer {self.__class__.__name__} does not have a `call()` "
            "method implemented."
        )

    def get_config(self):
        # 레이어의 구성 정보 반환
        
        config = {
            # 나중에 바꿔보자 이건
            'module': "dev.layers",
        }

        return {**config}

    @classmethod
    def from_config(cls, config):
        instance = cls(name=config['name'])
        instance.input_shape = config['input_shape']
        instance.output_shape = config['output_shape']
        return instance