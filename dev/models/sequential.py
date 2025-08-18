import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import sys, os
import json
import typing
import logging
import numpy as np
import cupy as cp

sys.path.append("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor/test")

from dev.layers.layer import Layer
from dev.backend.backend_ops.losses import losses as cuda_losses
from dev.backend.backend_ops.optimizers import optimizers
from dev.backend.backend_ops.metrics import metrics

# CUDA 연동 Pybind11 모듈
import graph_executor as ge
OpStruct = ge.OpStruct
Shape = ge.Shape

# 상수 정의
INPUT_ID = "input"

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Sequential:
    def __init__(self, layers=None, trainable=True, name=None, input_shape=None):
        self.built = False
        self._layers = []
        self.input_shape = input_shape
        self.trainable = trainable
        self.name = name
        self.output_var = None

        # 디바이스 상태
        self._device_ready = False     # 파라미터/옵티마이저 버퍼 준비 여부
        self.opt_buffers = {}          # {name: {"velocity": cp, "m": cp, "v": cp}}

        # 러닝 설정
        self.loss_type = None
        self.optimizer_type = None
        self.learning_rate = None

        # 그래프/셰이프
        self.E_raw = []
        self.E = []
        self.shapes = {}
        self.weights = {}
        self.biases = {}

        # 손실 출력 ID (compile에서 설정)
        self.loss_output_id = None

        # 글로벌 스텝
        self.global_step = 1

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

        # 레이어를 순회하며 그래프 구성
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

        # 모델 최종 출력(손실 이전) ID
        self.output_var = input_id

        # 학습 설정
        self.loss_type = loss
        self.optimizer_type = optimizer
        self.learning_rate = learning_rate

        # ✅ 마지막에 손실 노드 추가 (label은 param_id로)
        extra = ge.OpExtraParams()
        extra.label_id = "y_true"
        extra.loss_type = self.loss_type

        self.E_raw.append({
            "op_type": ge.OpType.LOSS,
            "input_id": self.output_var,   # 손실의 입력은 모델의 최종 출력
            "param_id": "y_true",
            "output_id": "loss",
            "extra_params": extra
        })
        # 🔥 역전파/손실 계산의 기준이 되는 출력 ID를 보관
        self.loss_output_id = "loss"

        self.built = True

        # ✅ OpStruct 변환
        self.E = []
        for op in self.E_raw:
            extra = op.get("extra_params", ge.OpExtraParams())
            param_id = op.get("param_id", "") or ""
            node = ge.OpStruct(
                ge.OpType(op["op_type"]),   # int -> enum 캐스팅
                str(op["input_id"]),
                str(param_id),
                str(op["output_id"]),
                extra
            )
            self.E.append(node)

    # 내부 유틸: 디바이스 파라미터/옵티마이저 버퍼 초기화(1회)
    def _ensure_device_state(self):
        if self._device_ready:
            return

        # 옵티마이저 상태 버퍼 준비 (필요 시 생성)
        for name in list(self.weights.keys()) + list(self.biases.keys()):
            param = self.weights.get(name) if name in self.weights else self.biases[name]
            if name not in self.opt_buffers:
                self.opt_buffers[name] = {
                    "velocity": cp.zeros_like(param),
                    "m": cp.zeros_like(param),
                    "v": cp.zeros_like(param)
                }

        self._device_ready = True

    # OptimizerType 매핑
    def _opt_type_enum(self):
        s = (self.optimizer_type or "sgd").lower()
        if s == "sgd":
            return ge.OptimizerType.SGD
        if s == "momentum":
            return ge.OptimizerType.MOMENTUM
        if s == "adam":
            return ge.OptimizerType.ADAM
        raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

    # Keras 스타일 한 배치 학습 (편의용)
    def train_on_batch(self, x, y):
        if not self.built:
            raise RuntimeError("✅ 모델이 컴파일되지 않았습니다. 먼저 compile()을 호출하세요.")
        self._ensure_device_state()

        x_cp = cp.asarray(x, dtype=cp.float32)
        y_cp = cp.asarray(y, dtype=cp.float32)
        batch_size_actual = x_cp.shape[0]

        # 입력/라벨 + 파라미터 포인터 맵
        tensor_ptrs = {
            "input": x_cp.data.ptr,
            "y_true": y_cp.data.ptr,
        }
        for name, arr in self.weights.items():
            tensor_ptrs[name] = arr.data.ptr
        for name, arr in self.biases.items():
            tensor_ptrs[name] = arr.data.ptr

        # 옵티마 상태 포인터 맵
        velocity_ptrs = {n: buf["velocity"].data.ptr for n, buf in self.opt_buffers.items()}
        m_ptrs        = {n: buf["m"].data.ptr        for n, buf in self.opt_buffers.items()}
        v_ptrs        = {n: buf["v"].data.ptr        for n, buf in self.opt_buffers.items()}

        loss_val = ge.train_step_entry(
            E=self.E,
            tensors=tensor_ptrs,
            shapes=self.shapes,
            # 🔥 역전파 시작점은 손실 출력
            final_output_id=(self.loss_output_id or self.output_var),
            label_tensor_id="y_true",
            loss_type=self.loss_type,
            batch_size=batch_size_actual,
            opt_type=self._opt_type_enum(),
            lr=self.learning_rate,
            beta1=0.9, beta2=0.999, eps=1e-8,
            timestep=self.global_step,
            velocity_ptrs=velocity_ptrs,
            m_ptrs=m_ptrs,
            v_ptrs=v_ptrs
        )

        self.global_step += 1
        return float(loss_val)

    def fit(self, x=None, y=None, epochs=1, batch_size=-1, verbose=0):
        if not self.built:
            raise RuntimeError("✅ 모델이 컴파일되지 않았습니다. 먼저 compile()을 호출하세요.")

        if batch_size == -1 or batch_size < x.shape[0]:
            batch_size = x.shape[0]

        self._ensure_device_state()

        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0

            indices = np.random.permutation(x.shape[0])
            x = x[indices]
            y = y[indices]

            x_cp = cp.asarray(x, dtype=cp.float32)
            y_cp = cp.asarray(y, dtype=cp.float32)

            for i in range(0, x_cp.shape[0], batch_size):
                batch_x = x_cp[i:i + batch_size]
                batch_y = y_cp[i:i + batch_size]

                loss_val = self.train_on_batch(batch_x, batch_y)

                epoch_loss += float(loss_val)
                batch_count += 1

            if verbose and batch_count > 0:
                avg_loss = epoch_loss / batch_count
                logger.info(f"[Epoch {epoch + 1}] 평균 손실: {avg_loss:.6f}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.built:
            raise RuntimeError("✅ 모델이 컴파일되지 않았습니다. 먼저 compile()을 호출하세요.")

        self._ensure_device_state()

        x_cp = cp.asarray(x, dtype=cp.float32)
        batch_size = x_cp.shape[0]

        # 입력 + 파라미터 포인터
        tensor_ptrs = {"input": x_cp.data.ptr}
        for name, arr in self.weights.items():
            tensor_ptrs[name] = arr.data.ptr
        for name, arr in self.biases.items():
            tensor_ptrs[name] = arr.data.ptr

        # 최종(손실 이전) 출력 shape
        out_shape = self.shapes[self.output_var]
        output_host = np.zeros((batch_size, out_shape.cols), dtype=np.float32)

        ge.run_graph_forward_entry(
            E=self.E,
            tensors=tensor_ptrs,
            shapes=self.shapes,
            out_host=output_host,
            final_output_id=self.output_var,   # predict는 손실 이전 출력이 필요
            batch_size=batch_size
        )
        return output_host

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        if not self.built:
            raise RuntimeError("✅ 모델이 컴파일되지 않았습니다. 먼저 compile()을 호출하세요.")

        self._ensure_device_state()

        x_cp = cp.asarray(x, dtype=cp.float32)
        y_cp = cp.asarray(y, dtype=cp.float32)
        batch_size = x_cp.shape[0]

        tensor_ptrs = {"input": x_cp.data.ptr, "y_true": y_cp.data.ptr}
        for name, arr in self.weights.items():
            tensor_ptrs[name] = arr.data.ptr
        for name, arr in self.biases.items():
            tensor_ptrs[name] = arr.data.ptr

        # 🔥 손실 기준으로 fwd+loss 수행
        loss_val = ge.run_graph_with_loss_entry(
            E=self.E,
            tensors=tensor_ptrs,
            shapes=self.shapes,
            final_output_id=(self.loss_output_id or self.output_var),
            label_tensor_id="y_true",
            loss_type=self.loss_type,
            batch_size=batch_size
        )
        return float(loss_val)
