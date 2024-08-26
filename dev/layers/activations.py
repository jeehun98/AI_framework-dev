from dev.layers.layer import Layer
from dev.layers import activations

# layer-Activation...

class Activation(Layer):
    def __init__(self, activation, **kwargs):
        super().__init__(**kwargs)
        self.activation = activations.get(activation)

    def call(self, inputs):
        return self.activation(inputs)
    
    def compute_output_shape(self, input_shape):
        return input_shape