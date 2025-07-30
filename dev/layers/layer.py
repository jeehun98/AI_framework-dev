class Layer:
    def __init__(self, name=None, regularizer=None, **kwargs):
        self.name = name or self.__class__.__name__
        self.input_shape = None
        self.output_shape = None
        self.built = False

        self.input_dim_arg = kwargs.pop("input_dim", None)
        if self.input_dim_arg is not None:
            self.input_dim_arg = (self.input_dim_arg,)

        """
        if regularizer is not None:
            self.regularizer = regularizers.get(regularizer)
        else:
            self.regularizer = None
        """
    def build(self, input_shape=None):
        if input_shape is not None:
            self.input_shape = input_shape
        elif self.input_dim_arg is not None:
            self.input_shape = self.input_dim_arg
        elif self.input_shape is not None:
            pass
        else:
            raise ValueError("Input shape must be provided during build.")
        
        self.built = True

        # 자동 output shape 계산 (있을 경우)
        if hasattr(self, "compute_output_shape"):
            self.output_shape = self.compute_output_shape(self.input_shape)

    def call(self, *args, **kwargs):
        if not self.built:
            raise RuntimeError(
                f"Layer '{self.name}' is not built yet. "
                "Please call `build()` before using this layer."
            )
        raise NotImplementedError(f"{self.__class__.__name__} has no call method.")

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def apply_regularizer(self, weights):
        if self.regularizer is not None:
            return self.regularizer(weights)
        return 0

    def get_config(self):
        return {
            'name': self.name,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'regularizer': self.regularizer.__class__.__name__ if self.regularizer else None,
            'module': "dev.layers",
        }

    @classmethod
    def from_config(cls, config):
        instance = cls(name=config['name'], regularizer=config.get('regularizer'))
        instance.input_shape = config.get('input_shape')
        instance.output_shape = config.get('output_shape')
        return instance
