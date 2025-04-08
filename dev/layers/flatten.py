import numpy as np
from functools import reduce
from operator import mul

from dev.layers.layer import Layer

class Flatten(Layer):
    def __init__(self, input_shape=None, **kwargs):
        super().__init__(input_shape, **kwargs)
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.trainable = False
        self.node_list = []
        self.layer_name = "flatten"

    def get_config(self):
        base_config = super().get_config()
        config = {
            "class_name": self.__class__.__name__,
            "input_shape": self.input_shape
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        elif inputs.ndim > 2:
            batch_size = inputs.shape[0]
            flattened_dim = int(np.prod(inputs.shape[1:]))
            inputs = inputs.reshape(batch_size, flattened_dim)
        
        self.output_shape = inputs.shape
        self.node_list = []
        return inputs


    def compute_output_shape(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:]))

    def multiply_tuple_elements(self, t):
        return reduce(mul, t, 1)

    def build(self, input_shape):
        result = self.multiply_tuple_elements(input_shape)
        self.output_shape = (1, result)
        super().build()
