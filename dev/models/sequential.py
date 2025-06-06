import typing
import json
import numpy as np
import cupy as cp

from dev.layers.layer import Layer
from dev.backend.backend_ops.losses import losses as cuda_losses
from dev.graph_engine.graph_compiler import GraphCompiler
from dev.backend.backend_ops.optimizers import optimizers
from dev import metrics

class Sequential:
    def __init__(self, layers=None, trainable=True, name=None, input_shape=None):
        self.built = False
        self._layers = []
        self.input_shape = input_shape
        self.trainable = trainable
        self.name = name
        self.graph_ops = []  # ✅ GraphCompiler용 연산 저장
        self.output_var = None  # ✅ 최종 출력 이름

    def get_config(self):
        sequential_config = {'name': 'sequential'}
        layer_configs = [layer.get_config() for layer in self._layers]
        return {**sequential_config, "layers": layer_configs}

    def add(self, layer):
        if not isinstance(layer, Layer):
            raise ValueError("Only instances of Layer can be added.")

        if self._layers:
            previous_layer = self._layers[-1]
            input_shape = previous_layer.output_shape
            if input_shape is not None:
                layer.build(input_shape)
        elif hasattr(layer, "input_shape") and layer.input_shape is not None:
            layer.build(layer.input_shape)
        else:
            raise RuntimeError("첫 번째 레이어는 input_shape를 지정해야 합니다.")

        print(f"✅ 레이어 추가됨: {layer.__class__.__name__} (input_shape={layer.input_shape}, output_shape={layer.output_shape})")
        self._layers.append(layer)

    def build(self):
        self.input_shape = self._layers[0].input_shape

    def get_build_config(self):
        return {"input_shape": self.input_shape}

    def compile(self, optimizer=None, loss=None, p_metrics=None, learning_rate=0.001):

        if optimizer is None:
            optimizer = 'sgd'
        if loss is None:
            loss = 'mse'
        if p_metrics is None:
            p_metrics = 'mse'

        self.compiler = GraphCompiler()

        for layer in self._layers:
            self.compiler.add_layer(layer)

        self.optimizer = optimizers.get(optimizer, learning_rate=learning_rate)
        self.loss_fn = cuda_losses.get(loss)
        self.loss_grad_fn = cuda_losses.get_grad(loss)
        self.loss_name = loss
        self.metric_fn = metrics.get(p_metrics)
        self.build()

    def get_compile_config(self):
        return {
            "optimizer": self.optimizer.get_config(),
            "loss": self.loss_fn.__name__,
            "metrics": self.metric_fn.get_config(),
        }

    def get_weight(self):
        self.weights = [layer.weights for layer in self._layers if hasattr(layer, 'weights')]

    def serialize_model(self):
        compile_config = self.get_compile_config()
        model_config = self.get_config()
        build_config = self.get_build_config()

        weights = []
        for layer in self._layers:
            layer_weights = layer.get_weights()
            serialized_weights = [w.tolist() for w in layer_weights]
            weights.append(serialized_weights)

        model_data = {
            "compile_config": compile_config,
            "model_config": model_config,
            "build_config": build_config,
            "weights": weights,
        }

        return json.dumps(model_data)

    def predict(self, data):
        output = data
        for i, layer in enumerate(self._layers):
            output = layer.call(output)
        return output

    def forward_pass(self, input_data):
        output = input_data
        for i, layer in enumerate(self._layers):
            output = layer.call(output)
        return output

    def compute_loss_and_metrics(self, y_pred, y_true):
        y_pred = cp.asarray(y_pred, dtype=cp.float32)
        y_true = cp.asarray(y_true, dtype=cp.float32)

        loss_value = self.loss_fn(y_true, y_pred)
        metric_value = self.metric_fn(y_pred, y_true)

        print("[DEBUG] y_pred:", y_pred)
        print("[DEBUG] target :", y_true)
        print("[DEBUG] loss fn input dtype:", y_pred.dtype)

        return loss_value, metric_value

    def backward_pass(self, grad_output):
        for layer in reversed(self._layers):
            grad_output = layer.backward(grad_output)

    def update_weights(self):
        for layer in self._layers:
            if hasattr(layer, "update"):
                layer.update(self.optimizer)

    def fit(self, x=None, y=None, epochs=1, batch_size=-1):
        if batch_size == -1 or batch_size < x.shape[0]:
            batch_size = x.shape[0]
        batch_counts = int(np.ceil(x.shape[0] / batch_size))

        for epoch in range(epochs):
            print(f"\n=== [Epoch {epoch + 1}] 시작 ===")

            indices = np.random.permutation(x.shape[0])
            x = x[indices]
            y = y[indices]

            x = cp.asarray(x, dtype=cp.float32)
            y = cp.asarray(y, dtype=cp.float32)

            for batch_idx in range(batch_counts):
                print(f"\n[Batch {batch_idx + 1}] 처리 시작")

                start = batch_idx * batch_size
                end = min(start + batch_size, x.shape[0])
                batch_x = x[start:end]
                batch_y = y[start:end]
                batch_datas = batch_x.shape[0]
                batch_loss_sum = 0

                for data_idx in range(batch_datas):
                    input_data = batch_x[data_idx:data_idx+1]
                    target = batch_y[data_idx:data_idx+1]

                    print(f"\n[SHAPE TRACE] === Sample {data_idx + 1} ===")
                    y_pred = self.forward_pass(input_data)

                    loss_value, metric_value = self.compute_loss_and_metrics(y_pred, target)
                    print(f"[DEBUG] 손실: {loss_value}, 메틱: {metric_value}")

                    target = cp.asarray(target, dtype=cp.float32)
                    y_pred = cp.asarray(y_pred, dtype=cp.float32)
                    grad = self.loss_grad_fn(target, y_pred)

                    self.backward_pass(grad)
                    batch_loss_sum += loss_value

                self.update_weights()
                batch_loss = batch_loss_sum / batch_datas
                print(f"[Batch {batch_idx + 1}] 평균 손실: {batch_loss}")

            total_loss = 0
            for i in range(x.shape[0]):
                pred = self.predict(x[i:i+1])
                loss, _ = self.compute_loss_and_metrics(pred, y[i:i+1])
                total_loss += loss

            print(f"\n📊 [Epoch {epoch + 1}] 전체 평균 손실: {total_loss / x.shape[0]}")

    # ✅ GraphCompiler용 연산 리스트 생성
    def compile_graph(self, input_var="x0"):
        self.graph_ops = []
        current_input = input_var

        for layer in self._layers:
            if hasattr(layer, "forward_matrix"):
                ops = layer.forward_matrix(current_input)
                self.graph_ops.extend(ops)
                current_input = ops[-1][-1]  # 마지막 연산의 output을 다음 input으로 설정

        self.output_var = current_input
        return self.graph_ops
