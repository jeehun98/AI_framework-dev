from dev.layers.layer import Layer
from dev import activations

# layer-Activation...

class Activation(Layer):
    # 초기화 시 설정되어야 할 부분, activation 값
    # Layer 를 상속받는 다양한 종류들, Dense, Activations ... 의 자식 레이어는
    # 각자, 특유의 파라미터를 입력으로 받으며 분리,
    # 해당 값을 attribute 로 저장하는 과정도 포함해야 한다.

    # add(Activation('sigmoid'))
    def __init__(self, activation, **kwargs):
        # 계속 부모 클래스로 이동해서 인스턴스를 생성해야 함
        super().__init__(**kwargs)
        self.activation = activations.get(activation)
        self.node_list = []
        self.trainable = True

    # call, 연산 수행시 실제 메서드가 위치하는 곳에서 연산 수행
    def call(self, inputs):
        output, activation_node_list = self.activation(inputs, self.activation)
        self.node_list = activation_node_list
        return output
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def build(self, input_shape):
        self.input_shape = input_shape
        super()