import typing
import json
import numpy as np

from dev.losses import losses_mapping
from dev import optimizers
from dev.backend.backend_ops.losses import losses as cuda_losses
from dev import metrics


class SequentialMat:
    def __init__(self, layers=None):
        self._layers = []
        self.loss = None
        self.loss_name = None
        self.metric = None
        self.optimizer = None
        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        if self._layers:
            prev_layer = self._layers[-1]
            layer.input_dim = prev_layer.output_dim
        if hasattr(layer, "build"):
            layer.build((1, layer.input_dim))
        self._layers.append(layer)
        print(f"✅ 레이어 추가됨: {layer.__class__.__name__} (input_dim={layer.input_dim}, output_dim={layer.output_dim})")

    def compile(self, optimizer=None, loss=None, p_metrics=None, learning_rate=0.001):
        self.optimizer = optimizers.get(optimizer, learning_rate=learning_rate)
        self.loss = cuda_losses.get(loss)
        self.loss_name = loss
        self.metric = metrics.get(p_metrics)

    def predict(self, x):
        output = x
        for layer in self._layers:
            output = layer.call(output)
        return output

    def compute_loss_and_metrics(self, y_pred, y_true):
        loss_val = self.loss(y_true, y_pred)
        metric_val = self.metric(y_pred, y_true)
        return loss_val, metric_val

    def fit(self, x, y, epochs=1, batch_size=-1):
        if batch_size == -1 or batch_size < x.shape[0]:
            batch_size = x.shape[0]
        num_batches = int(np.ceil(x.shape[0] / batch_size))

        for epoch in range(epochs):
            print(f"\n=== [Epoch {epoch + 1}] 시작 ===")
            indices = np.random.permutation(x.shape[0])
            x, y = x[indices], y[indices]
            
            for batch in range(num_batches):
                start = batch * batch_size
                end = min(start + batch_size, x.shape[0])
                batch_x, batch_y = x[start:end], y[start:end]
                batch_loss = 0

                for i in range(batch_x.shape[0]):
                    xi = batch_x[i].reshape(1, -1)
                    yi = batch_y[i].reshape(1, -1)

                    # Forward
                    output = self.predict(xi)

                    # Loss
                    loss_val, metric_val = self.compute_loss_and_metrics(output, yi)
                    batch_loss += loss_val

                    # TODO: Backward + Weight update for DenseMat
                    # self.backward_and_update(xi, yi, output)

                print(f"[Batch {batch + 1}] 평균 손실: {batch_loss / batch_x.shape[0]}")
