import typing
import json
import numpy as np
import cupy as cp
import sys
import os

from dev.layers.layer import Layer
from dev.backend.backend_ops.losses import losses as cuda_losses
from dev.backend.backend_ops.optimizers import optimizers
from dev import metrics

# CUDA 연동 Pybind11 모듈
import graph_executor as ge
OpStruct = ge.OpStruct
Shape = ge.Shape


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
                layer.build(input_shape)
        elif hasattr(layer, "input_shape") and layer.input_shape is not None:
            layer.build(layer.input_shape)
        else:
            raise RuntimeError("첫 번째 레이어는 input_shape를 지정해야 합니다.")

        self._layers.append(layer)
        print(f"✅ 레이어 추가됨: {layer.__class__.__name__} (input_shape={layer.input_shape}, output_shape={layer.output_shape})")

    def compile(self, optimizer='sgd', loss='mse', p_metrics='mse', learning_rate=0.001):
        self.E_raw = []
        self.weights = {}
        self.biases = {}
        self.shapes = {}

        current_shape = None
        input_id = "input"

        for i, layer in enumerate(self._layers):
            if i == 0:
                if not layer.input_shape:
                    raise ValueError("첫 번째 레이어에 input_shape가 필요합니다.")
                current_shape = layer.input_shape

            layer.build(current_shape)
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
        np.savez("compiled_graph.npz", E=self.E_raw, weights=self.weights, biases=self.biases)

    def run_forward(self, input_data: np.ndarray):
        E = [OpStruct(int(op["op_type"]), str(op["input_id"]), str(op["param_id"]) if op["param_id"] else "", str(op["output_id"])) for op in self.E_raw]

        cp_input = cp.asarray(input_data, dtype=cp.float32)
        tensor_ptrs = {"input": cp_input.data.ptr}

        self.tensor_map = {"input": cp_input.copy()}  # tensor_map 초기화 및 저장

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

        ge.run_graph_cuda(E, tensor_ptrs, self.shapes, out_host, final_output_id=self.output_var)
        self.E = E  # run_backward용 저장
        return out_host
    
    def run_backward(self, x, grad_y):
        from cupy.cuda import MemoryPointer, UnownedMemory

        # 1. forward 실행 시 사용된 tensor_map을 기반으로 포인터 구성
        tensor_ptrs = {name: arr.data.ptr for name, arr in self.tensor_map.items()}

        # 2. grad_y는 최종 출력 변수에 대한 gradient
        grad_ptrs = {self.output_var: grad_y.data.ptr}

        # 3. 중간 연산 결과의 grad 버퍼도 미리 초기화하여 등록해야 함
        self.grad_buffers = {}  # 중간 gradient를 저장하는 버퍼
        for op in reversed(self.E):
            out_id = op.output_id
            if out_id not in grad_ptrs:
                shape = self.shapes[out_id]
                grad_buf = cp.zeros((shape.rows, shape.cols), dtype=cp.float32)
                grad_ptrs[out_id] = grad_buf.data.ptr
                self.grad_buffers[out_id] = grad_buf

        # 4. CUDA backward 실행
        grad_map = ge.run_graph_backward(self.E, tensor_ptrs, self.shapes, grad_ptrs, final_output_id=self.output_var)

        # 5. 안전하게 gradients 복원 (weights, biases에 대해)
        grads = {}
        for name in list(self.weights.keys()) + list(self.biases.keys()):
            ptr = grad_map.get(name, 0)
            if ptr != 0:
                shape = (self.shapes[name].rows, self.shapes[name].cols)
                size_bytes = shape[0] * shape[1] * 4  # float32 = 4 bytes
                mem = UnownedMemory(ptr, size_bytes, owner=None)
                memptr = MemoryPointer(mem, 0)
                grads[name] = cp.ndarray(shape, dtype=cp.float32, memptr=memptr).copy()
            else:
                print(f"⚠️ Gradient ptr for {name} is NULL")

        return grads



    def fit(self, x=None, y=None, epochs=1, batch_size=-1):
        if batch_size == -1 or batch_size < x.shape[0]:
            batch_size = x.shape[0]

        for epoch in range(epochs):
            print(f"\n=== [Epoch {epoch + 1}] 시작 ===")
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
                    print(f"[INFO] 손실: {loss:.6f}")

                    grad_y = self.loss_grad_fn(target, y_pred)
                    grad_map = self.run_backward(input_data, grad_y)

                    for name in self.weights:
                        if name in grad_map:
                            self.weights[name] -= self.optimizer.lr * grad_map[name]
                    for name in self.biases:
                        if name in grad_map:
                            self.biases[name] -= self.optimizer.lr * grad_map[name]

                    batch_loss_sum += loss

                print(f"[Batch 완료] 평균 손실: {batch_loss_sum / batch_x.shape[0]:.6f}")
