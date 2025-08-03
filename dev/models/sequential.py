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
from dev.backend.backend_ops.metrics import metrics

sys.path.append("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor/test")

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
        self.metric_type = p_metrics
        self.shapes = {}

        input_id = INPUT_ID
        current_shape = self.input_shape

        for i, layer in enumerate(self._layers):
            if i == 0:
                if not layer.input_shape:
                    raise ValueError("첫 번째 레이어에 input_shape가 필요합니다.")
                layer.build(layer.input_shape)
                current_shape = layer.compute_output_shape(layer.input_shape)
            else:
                layer.input_shape = current_shape
                layer.build(current_shape)
                current_shape = layer.compute_output_shape(current_shape)

            e_block, w, b, output_id, shape_map = layer.to_e_matrix(input_id)
            self.E_raw.extend(e_block)
            self.weights.update(w)
            self.biases.update(b)
            self.shapes.update(shape_map)
            input_id = output_id

        self.output_var = input_id

        # ✅ CUDA 기반으로 전달된 문자열 저장 (실제 연산은 CUDA에서 처리)
        self.loss_type = loss
        self.optimizer_type = optimizer
        self.learning_rate = learning_rate

        self.built = True

        # compile() 끝부분에서 E_raw → E 변환 추가
        self.E = []
        for op in self.E_raw:
            extra = op.get("extra_params", ge.OpExtraParams())
            param_id = op.get("param_id", "") or ""
            node = ge.OpStruct(
                int(op["op_type"]),
                str(op["input_id"]),
                str(param_id),
                str(op["output_id"]),
                extra
            )
            self.E.append(node)


    def run_forward(self, input_data: np.ndarray):

        try:
            E = []
            for op in self.E_raw:
                extra = op.get("extra_params", ge.OpExtraParams())
                param_id = op.get("param_id", "")
                if param_id is None:
                    param_id = ""
                node = OpStruct(
                    int(op["op_type"]),
                    str(op["input_id"]),
                    str(param_id),
                    str(op["output_id"]),
                    extra
                )
                E.append(node)

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
            ge.run_graph_forward_entry(
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
            grad_map = ge.run_graph_backward_entry(
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

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.built:
            raise RuntimeError("✅ 모델이 컴파일되지 않았습니다. 먼저 compile()을 호출하세요.")

        x_cp = cp.asarray(x, dtype=cp.float32)
        batch_size = x_cp.shape[0]

        tensor_ptrs = {"input": x_cp.data.ptr}
        self.tensor_map = {"input": x_cp.copy()}

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
                buf = cp.empty((shape.rows, shape.cols), dtype=cp.float32)
                tensor_ptrs[var] = buf.data.ptr
                self.tensor_map[var] = buf

        output_host = np.zeros((self.shapes[self.output_var].rows, self.shapes[self.output_var].cols), dtype=np.float32)

        ge.run_graph_forward_entry(
            E=self.E,
            tensors=tensor_ptrs,
            shapes=self.shapes,
            out_host=output_host,
            final_output_id=self.output_var,
            batch_size=batch_size
        )

        return output_host



    def fit(self, x=None, y=None, epochs=1, batch_size=-1):
        if batch_size == -1 or batch_size < x.shape[0]:
            batch_size = x.shape[0]

        self.global_step = 1  # Adam 등에서 필요한 timestep

        for epoch in range(epochs):
            logger.info(f"\n=== [Epoch {epoch + 1}] 시작 ===")
            indices = np.random.permutation(x.shape[0])
            x = x[indices]
            y = y[indices]

            x_cp = cp.asarray(x, dtype=cp.float32)
            y_cp = cp.asarray(y, dtype=cp.float32)

            for i in range(0, x_cp.shape[0], batch_size):
                batch_x = x_cp[i:i+batch_size]
                batch_y = y_cp[i:i+batch_size]
                batch_size_actual = batch_x.shape[0]

                tensor_ptrs = {"input": batch_x.data.ptr, "y_true": batch_y.data.ptr}
                self.tensor_map = {"input": batch_x.copy(), "y_true": batch_y.copy()}

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
                        buf = cp.empty((shape.rows, shape.cols), dtype=cp.float32)
                        tensor_ptrs[var] = buf.data.ptr
                        self.tensor_map[var] = buf

                # ✅ 손실 및 forward 실행
                loss_val = ge.run_graph_with_loss_entry(
                    E=self.E,
                    tensors=tensor_ptrs,
                    shapes=self.shapes,
                    final_output_id=self.output_var,
                    label_tensor_id="y_true",
                    loss_type=self.loss_type,
                    batch_size=batch_size_actual
                )

                # ✅ 역전파 실행
                grad_ptrs = {self.output_var: 0}  # 시작점 (자동 초기화됨)
                grad_map = ge.run_graph_backward_entry(
                    E=self.E,
                    tensors=tensor_ptrs,
                    shapes=self.shapes,
                    gradients=grad_ptrs,
                    final_output_id=self.output_var,
                    batch_size=batch_size_actual
                )

                # ✅ CUDA 기반 Optimizer 적용
                for name in list(self.weights.keys()) + list(self.biases.keys()):
                    param = self.tensor_map[name]
                    grad_ptr = grad_map.get(name, 0)
                    if grad_ptr == 0:
                        continue

                    grad = cp.ndarray(param.shape, dtype=cp.float32,
                                    memptr=cp.cuda.MemoryPointer(
                                        cp.cuda.UnownedMemory(grad_ptr, param.nbytes, None), 0))

                    # 옵티마이저 상태 버퍼 준비
                    if not hasattr(self, "opt_buffers"):
                        self.opt_buffers = {}

                    if name not in self.opt_buffers:
                        self.opt_buffers[name] = {
                            "velocity": cp.zeros_like(param),
                            "m": cp.zeros_like(param),
                            "v": cp.zeros_like(param)
                        }

                    velocity = self.opt_buffers[name]["velocity"]
                    m = self.opt_buffers[name]["m"]
                    v = self.opt_buffers[name]["v"]


                    # 1. 먼저 optimizer_type 문자열을 소문자로 정리
                    opt_type_str = self.optimizer_type.lower()

                    # 2. 문자열에 따라 올바른 enum 값 선택
                    if opt_type_str == "sgd":
                        opt_type_enum = ge.OptimizerType.SGD
                    elif opt_type_str == "momentum":
                        opt_type_enum = ge.OptimizerType.MOMENTUM
                    elif opt_type_str == "adam":
                        opt_type_enum = ge.OptimizerType.ADAM
                    else:
                        raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

                    ge.optimizer_update(
                        param_ptr=param.data.ptr,
                        grad_ptr=grad.data.ptr,
                        velocity_ptr=velocity.data.ptr,
                        m_ptr=m.data.ptr,
                        v_ptr=v.data.ptr,
                        lr=self.learning_rate,
                        beta1=0.9,
                        beta2=0.999,
                        eps=1e-8,
                        size=param.size,
                        opt_type=opt_type_enum,
                        timestep=self.global_step
                    )

                    # 업데이트된 파라미터 반영
                    self.weights[name] = param
                    self.biases[name] = param if name in self.biases else self.biases.get(name)

                self.global_step += 1

                logger.info(f"[Batch 완료] 손실: {loss_val:.6f}")



    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        if not self.built:
            raise RuntimeError("✅ 모델이 컴파일되지 않았습니다. 먼저 compile()을 호출하세요.")

        x_cp = cp.asarray(x, dtype=cp.float32)
        y_cp = cp.asarray(y, dtype=cp.float32)
        batch_size = x_cp.shape[0]

        tensor_ptrs = {
            "input": x_cp.data.ptr,
            "y_true": y_cp.data.ptr
        }

        self.tensor_map = {
            "input": x_cp.copy(),
            "y_true": y_cp.copy()
        }

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
                buf = cp.empty((shape.rows, shape.cols), dtype=cp.float32)
                tensor_ptrs[var] = buf.data.ptr
                self.tensor_map[var] = buf

        loss_val = ge.run_graph_with_loss_entry(
            E=self.E,
            tensors=tensor_ptrs,
            shapes=self.shapes,
            final_output_id=self.output_var,
            label_tensor_id="y_true",
            loss_type=self.loss_type,
            batch_size=batch_size
        )

            # ✅ Metric 계산
        output_arr = cp.asarray(self.tensor_map[self.output_var])
        y_true_arr = cp.asarray(self.tensor_map["y_true"])

        if self.metric_type.lower() == "mse":
            metric_result = metrics.mse(output_arr, y_true_arr)
        elif self.metric_type.lower() == "mae":
            metric_result = metrics.mae(output_arr, y_true_arr)
        elif self.metric_type.lower() == "accuracy":
            metric_result = metrics.accuracy(output_arr, y_true_arr)
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")

        logger.info(f"📊 평가 손실: {loss_val:.6f}, 메트릭({self.metric_type}): {metric_result:.6f}")
        return float(metric_result)