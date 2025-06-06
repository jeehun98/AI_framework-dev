from dev.layers.layer import Layer
import cupy as cp

class MaxPooling2D(Layer):
    def __init__(self, pool_size=2, stride=2, name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride

    def call(self, input_data):  # input_data: (batch, height, width, channels)
        input_data = cp.asarray(input_data, dtype=cp.float32)
        self.input_shape = input_data.shape
        batch, in_h, in_w, channels = self.input_shape
        ph, pw = self.pool_size
        sh = sw = self.stride

        out_h = (in_h - ph) // sh + 1
        out_w = (in_w - pw) // sw + 1
        output = cp.zeros((batch, out_h, out_w, channels), dtype=cp.float32)

        for b in range(batch):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start, h_end = i * sh, i * sh + ph
                        w_start, w_end = j * sw, j * sw + pw
                        region = input_data[b, h_start:h_end, w_start:w_end, c]
                        output[b, i, j, c] = cp.max(region)

        return output

    def backward(self, grad_output):
        raise NotImplementedError("MaxPooling2D.backward() is not implemented.")
