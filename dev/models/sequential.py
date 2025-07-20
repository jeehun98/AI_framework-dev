import typing
import json
import numpy as np
import cupy as cp
import sys
import os
import logging

from dev.layers.layer import Layer
from dev.backend.backend_ops.losses import losses as cuda_losses
from dev.backend.backend_ops.optimizers import optimizers
from dev import metrics

sys.path.append("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor")

# CUDA 연동 Pybind11 모듈
import graph_executor as ge
OpStruct = ge.OpStruct
Shape = ge.Shape

# 상수 정의
INPUT_ID = "input"
GRAPH_FILE = "compiled_graph.npz"

# 로깅 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Sequential:
    def __init__(self, layers=None, trainable=True, name=None, input_shape=None):
        self.built = False
        self._layers = []
        self.input_shape = input_shape
        self.trainable = trainable
        self.name = name
        self.output_var = None

    def add(self, layer):
        if not isinstance(layer, Layer):
            raise ValueError("Only instances of Layer can be added.")
        if self._layers:
            prev = self._layers[-1]
            input_shape = prev.output_shape
            if input_shape:
                layer.input_shape = input_shape
                layer.build(input_shape)
        elif layer.input_shape is not None:
            layer.build(layer.input_shape)
        else:
            raise RuntimeError("첫 번째 레이어는 input_shape를 지정해야 합니다.")
        self._layers.append(layer)
        logger.info(f"✅ 레이어 추가됨: {layer.__class__.__name__} (input_shape={layer.input_shape}, output_shape={layer.output_shape})")

    def compile(self, optimizer='sgd', loss='mse', p_metrics='mse', learning_rate=0.001):
        self.E_raw = []
        self.weights = {}
        self.biases = {}
        self.shapes = {}

        input_id = INPUT_ID
        current_shape = self.input_shape

        for i, layer in enumerate(self._layers):
            if i == 0 and not layer.input_shape:
                raise ValueError("첫 번째 레이어에 input_shape가 필요합니다.")
            current_shape = layer.compute_output_shape(current_shape)
            e_block, w, b, output_id, shape_map = layer.to_e_matrix(input_id)
            self.E_raw.extend(e_block)
            self.weights.update(w)
            self.biases.update(b)
            self.shapes.update(shape_map)
            input_id = output_id

        self.output_var = input_id

        self.optimizer = optimizers.get(optimizer, learning_rate=learning_rate)
        self.loss_fn = cuda_losses.get(loss)
        self.loss_grad_fn = cuda_losses.get_grad(loss)
        self.metric_fn = metrics.get(p_metrics)

        self.built = True
        np.savez(GRAPH_FILE, E=self.E_raw, weights=self.weights, biases=self.biases)

    def run_forward(self, input_data: np.ndarray):
        try:
            E = [OpStruct(int(op["op_type"]), str(op["input_id"]),
                          str(op["param_id"]) if op["param_id"] else "",
                          str(op["output_id"])) for op in self.E_raw]
        except Exception as e:
            raise RuntimeError(f"연산 구조 생성 실패: {e}")

        cp_input = cp.asarray(input_data, dtype=cp.float32)
        tensor_ptrs = {INPUT_ID: cp_input.data.ptr}
        self.tensor_map = {INPUT_ID: cp_input.copy()}

        for name, arr in self.weights.items():
            cp_arr = cp.asarray(arr, dtype=cp.float32)
            tensor_ptrs[name] = cp_arr.data.ptr
            self.tensor_map[name] = cp_arr.copy()

        for name, arr in self.biases.items():
            cp_arr = cp.asarray(arr, dtype=cp.float32)
            tensor_ptrs[name] = cp_arr.data.ptr
            self.tensor_map[name] = cp_arr.copy()

        for var, shape in self.shapes.items():
            if var not in tensor_ptrs:
                cp_arr = cp.empty((shape.rows, shape.cols), dtype=cp.float32)
                tensor_ptrs[var] = cp_arr.data.ptr
                self.tensor_map[var] = cp_arr

        out_shape = (self.shapes[self.output_var].rows, self.shapes[self.output_var].cols)
        out_host = np.zeros(out_shape, dtype=np.float32)

        try:
            ge.run_graph_cuda(
                E,
                tensor_ptrs,
                self.shapes,
                out_host,
                final_output_id=self.output_var,
                batch_size=input_data.shape[0]  # ✅ 여기 추가
            )
        except Exception as e:
            raise RuntimeError(f"CUDA forward execution failed: {e}")

        self.E = E
        return out_host

    def run_backward(self, x, grad_y):
        from cupy.cuda import MemoryPointer, UnownedMemory

        if not hasattr(self, "tensor_map"):
            raise RuntimeError("tensor_map is not initialized. run_forward() must be called first.")

        tensor_ptrs = {name: arr.data.ptr for name, arr in self.tensor_map.items()}
        grad_ptrs = {self.output_var: grad_y.data.ptr}
        self.grad_buffers = {}

        for op in reversed(self.E):
            out_id = op.output_id
            if out_id not in grad_ptrs:
                shape = self.shapes[out_id]
                grad_buf = cp.zeros((shape.rows, shape.cols), dtype=cp.float32)
                grad_ptrs[out_id] = grad_buf.data.ptr
                self.grad_buffers[out_id] = grad_buf

        try:
            grad_map = ge.run_graph_backward(
            self.E,
            tensor_ptrs,
            self.shapes,
            grad_ptrs,
            self.output_var,
            batch_size=x.shape[0]   # ✅ 이 줄 추가
        )

        except Exception as e:
            raise RuntimeError(f"CUDA backward execution failed: {e}")

        grads = {}
        for name in list(self.weights.keys()) + list(self.biases.keys()):
            ptr = grad_map.get(name, 0)
            if ptr != 0:
                shape = (self.shapes[name].rows, self.shapes[name].cols)
                size_bytes = shape[0] * shape[1] * 4
                mem = UnownedMemory(ptr, size_bytes, owner=None)
                memptr = MemoryPointer(mem, 0)
                grads[name] = cp.ndarray(shape, dtype=cp.float32, memptr=memptr).copy()
            else:
                logger.warning(f"⚠️ Gradient ptr for {name} is NULL")

        return grads

    def fit(self, x=None, y=None, epochs=1, batch_size=-1):
        if batch_size == -1 or batch_size < x.shape[0]:
            batch_size = x.shape[0]

        for epoch in range(epochs):
            logger.info(f"\n=== [Epoch {epoch + 1}] 시작 ===")
            indices = np.random.permutation(x.shape[0])
            x = x[indices]
            y = y[indices]

            x = cp.asarray(x, dtype=cp.float32)
            y = cp.asarray(y, dtype=cp.float32)

            for i in range(0, x.shape[0], batch_size):
                batch_x = x[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                batch_loss_sum = 0

                for j in range(batch_x.shape[0]):
                    input_data = batch_x[j:j+1]
                    target = batch_y[j:j+1]

                    y_pred = self.run_forward(input_data)
                    loss = self.loss_fn(target, y_pred)
                    metric_value = self.metric_fn(target, y_pred)
                    logger.info(f"[INFO] 손실: {loss:.6f}, 메트릭: {metric_value:.6f}")

                    grad_y = self.loss_grad_fn(target, y_pred)
                    grad_map = self.run_backward(input_data, grad_y)

                    for name in self.weights:
                        if name in grad_map:
                            self.weights[name] -= self.optimizer.lr * grad_map[name]
                    for name in self.biases:
                        if name in grad_map:
                            self.biases[name] -= self.optimizer.lr * grad_map[name]

                    batch_loss_sum += loss

                avg_loss = batch_loss_sum / batch_x.shape[0]
                logger.info(f"[Batch 완료] 평균 손실: {avg_loss:.6f}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.built:
            raise RuntimeError("✅ 모델이 컴파일되지 않았습니다. 먼저 compile()을 호출하세요.")

        x_cp = cp.asarray(x, dtype=cp.float32)
        output = self.run_forward(x_cp)
        return output
