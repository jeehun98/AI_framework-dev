import warnings

from layers.layer import Layer

class InputLayer(Layer):
    def __init__(
        self,
        shape=None,
        name=None,
        **kwargs,
    ):
        # TODO: support for ragged.
        super().__init__(name=name)
        