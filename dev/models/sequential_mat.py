import typing
import json
import numpy as np


from dev.graph_engine.graph_compiler import GraphCompiler
from dev.losses import losses_mapping
from dev import optimizers
from dev.backend.backend_ops.losses import losses as cuda_losses
from dev import metrics
from dev.backend.backend_ops.activations import activations as cuda_activations
from dev.layers.dense_mat import DenseMat
from dev.layers.activation_mat import ActivationMat

from ..tests.test_setup import import_cuda_module

matrix_ops = import_cuda_module(
    module_name="operations_matrix_cuda",
    build_dir=r"C:/Users/owner/Desktop/AI_framework-dev/dev/backend/backend_ops/operaters/build/lib.win-amd64-cpython-312"
)


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
            prev_output_dim = self._layers[-1].output_dim
            layer.build(prev_output_dim)
        elif hasattr(layer, 'input_dim') and layer.input_dim is not None:
            layer.build(layer.input_dim)
        else:
            raise ValueError("첫 번째 레이어는 input_dim이 필요합니다.")
        self._layers.append(layer)

    def compile(self, optimizer=None, loss=None, p_metrics=None, learning_rate=0.001):
        self.optimizer = optimizers.get(optimizer, learning_rate=learning_rate)
        self.loss = cuda_losses.get(loss)
        self.loss_name = loss
        self.metric = metrics.get(p_metrics)

        # ✅ 기존 plan 기반 forward 연산 구성
        self.forward_plan = []
        for i, layer in enumerate(self._layers):
            if hasattr(layer, "forward_matrix"):
                plan = layer.forward_matrix()
                self.forward_plan.append(plan)
                print(f"[DEBUG] forward_plan[{i}]:", plan)
            else:
                print(f"[WARN] 레이어 {layer} 는 forward_matrix() 미구현 → forward_plan 생략")

        # ✅ 그래프 컴파일러를 통한 행렬 기반 연산 준비
        self.graph_compiler = GraphCompiler()

        for layer in self._layers:
            self.graph_compiler.add_layer(layer)
        self.graph_compiler.build()

        matrices = self.graph_compiler.get_matrices()
        print(f"[DEBUG] op_matrix shape: {np.shape(matrices['op_matrix'])}")
        print(f"[DEBUG] input_matrix shape: {np.shape(matrices['input_matrix'])}")
        print(f"[DEBUG] param_vector length: {len(matrices['param_vector'])}")


    def predict(self, x):
        output = x
        for layer in self._layers:
            output = layer.call(output)
        return output

    def build_forward_plan(self, x):
        self.backward_plan = []
        output = x.astype(np.float32)

        for layer in self._layers:
            if isinstance(layer, DenseMat):
                layer_input = output.copy()
                output = layer.call(output)

                self.backward_plan.append({
                    "type": "dense",
                    "input": layer_input,
                    "weight": layer.weights.copy(),
                    "bias": layer.bias.copy(),
                })

            elif isinstance(layer, ActivationMat):
                layer_input = output.copy()
                output = layer.call(output)

                self.backward_plan.append({
                    "type": "activation",
                    "activation": layer.activation_name,
                    "input": layer_input,
                })

        return output


    def compute_loss_and_metrics(self, y_pred, y_true):
        loss_val = self.loss(y_true, y_pred)
        metric_val = self.metric(y_pred, y_true)
        return loss_val, metric_val

    def fast_forward(self, x):
        """
        forward_plan 기반으로 구성된 연산 계획을 따라 순전파를 한 번에 수행.
        각 레이어를 직접 호출하지 않고, plan 정보만을 사용.
        """
        output = np.atleast_2d(x).astype(np.float32)

        for i, plan in enumerate(self.forward_plan):
            if plan["type"] == "dense":
                # ✅ Dense 연산
                W = plan["weight"]
                b = plan["bias"]
                try:
                    z = matrix_ops.matrix_multiply(output, W)
                    if isinstance(z, tuple):
                        z = z[0]
                except Exception:
                    z = np.dot(output, W)

                bias_tiled = np.tile(b, (output.shape[0], 1))
                try:
                    z = matrix_ops.matrix_add(z, bias_tiled)
                    if isinstance(z, tuple):
                        z = z[0]
                except Exception:
                    z = z + bias_tiled

                output = z  # 덮어쓰기

            elif plan["type"] == "activation":
                # ✅ Activation 연산
                act_name = plan["activation"]
                act_func = {
                    "relu": cuda_activations.relu,
                    "sigmoid": cuda_activations.sigmoid,
                    "tanh": cuda_activations.tanh,
                }.get(act_name)

                if act_func is None:
                    raise NotImplementedError(f"[ERROR] 활성화 함수 '{act_name}' 지원되지 않음")

                output = act_func(output)

            else:
                raise ValueError(f"[ERROR] 알 수 없는 레이어 타입 '{plan['type']}'")

        return output

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

                    # ✅ 고속 순전파 수행
                    output = self.fast_forward(xi)

                    # 손실 및 메트릭 계산
                    loss_val, metric_val = self.compute_loss_and_metrics(output, yi)
                    batch_loss += loss_val

                    # ⚠️ 역전파 & 업데이트 로직은 추후 별도 구현 필요
                    # self.backward_and_update(...)

                print(f"[Batch {batch + 1}] 평균 손실: {batch_loss / batch_x.shape[0]}")

    def fast_backward(self, y_pred, y_true):
        grad = self.loss.grad(y_true, y_pred).astype(np.float32)  # dL/dy

        for plan in reversed(self.backward_plan):
            if plan["type"] == "activation":
                z = plan["input"].astype(np.float32)
                grad = self._activation_backward(plan["activation"], z, grad)

            elif plan["type"] == "dense":
                A = plan["input"].astype(np.float32)
                W = plan["weight"].astype(np.float32)

                # dW = A.T @ grad
                dW = matrix_ops.matrix_multiply(A.T, grad)
                # dA = grad @ W.T
                dA = matrix_ops.matrix_multiply(grad, W.T)
                # db = grad.sum(axis=0, keepdims=True)
                db = np.sum(grad, axis=0, keepdims=True)

                plan["grad_W"] = dW
                plan["grad_b"] = db
                grad = dA  # propagate to next layer
