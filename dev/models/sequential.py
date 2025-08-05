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

        # ✅ CUDA 런타임용 학습 설정 저장
        self.loss_type = loss
        self.optimizer_type = optimizer
        self.learning_rate = learning_rate

        # ✅ 마지막에 손실 노드 추가 (label은 param_id에 담아줌)
        extra = ge.OpExtraParams()
        extra.label_id = "y_true"  # ✅ 🔥 핵심 추가
        extra.loss_type = self.loss_type

        self.E_raw.append({
            "op_type": ge.OpType.LOSS,
            "input_id": self.output_var,
            "param_id": "y_true",
            "output_id": "loss",
            "extra_params": extra     # 🔑 이걸 같이 넣어야 Pybind11을 통해 전달됨
        })


        self.built = True

        # ✅ OpStruct 변환
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

    def fit(self, x=None, y=None, epochs=1, batch_size=-1):
        if batch_size == -1 or batch_size < x.shape[0]:
            batch_size = x.shape[0]

        self.global_step = 1  # Adam 등에서 필요한 timestep

        # 옵티마이저 상태 버퍼 초기화
        if not hasattr(self, "opt_buffers"):
            self.opt_buffers = {}

        for epoch in range(epochs):
            logger.info(f"\n=== [Epoch {epoch + 1}] 시작 ===")

            indices = np.random.permutation(x.shape[0])
            x = x[indices]
            y = y[indices]

            x_cp = cp.asarray(x, dtype=cp.float32)
            y_cp = cp.asarray(y, dtype=cp.float32)

            for i in range(0, x_cp.shape[0], batch_size):
                batch_x = x_cp[i:i + batch_size]
                batch_y = y_cp[i:i + batch_size]
                batch_size_actual = batch_x.shape[0]

                tensor_ptrs = {"input": batch_x.data.ptr, "y_true": batch_y.data.ptr}
                self.tensor_map = {"input": batch_x.copy(), "y_true": batch_y.copy()}

                # 가중치 및 편향 포인터 등록
                for name, arr in self.weights.items():
                    cp_arr = cp.asarray(arr, dtype=cp.float32)
                    tensor_ptrs[name] = cp_arr.data.ptr
                    self.tensor_map[name] = cp_arr.copy()

                for name, arr in self.biases.items():
                    cp_arr = cp.asarray(arr, dtype=cp.float32)
                    tensor_ptrs[name] = cp_arr.data.ptr
                    self.tensor_map[name] = cp_arr.copy()

                # 중간 변수용 메모리 확보
                for var, shape in self.shapes.items():
                    if var not in tensor_ptrs:
                        buf = cp.empty((shape.rows, shape.cols), dtype=cp.float32)
                        tensor_ptrs[var] = buf.data.ptr
                        self.tensor_map[var] = buf

                # ✅ Forward + Loss
                loss_val = ge.run_graph_with_loss_entry(
                    E=self.E,
                    tensors=tensor_ptrs,
                    shapes=self.shapes,
                    final_output_id="loss",        # 🔵 손실 노드 ID 명시
                    label_tensor_id="y_true",
                    loss_type=self.loss_type,
                    batch_size=batch_size_actual
                )

                # ✅ 1. 손실 노드에 대한 초기 gradient (dL/dL = 1)
                loss_grad = cp.array([1.0], dtype=cp.float32)

                # ✅ 2. 역전파 시작은 "loss" 노드로부터
                grad_ptrs = {
                    "loss": loss_grad.data.ptr
                }

                # ✅ 3. Backward 실행 (output_id = "loss")
                grad_map = ge.run_graph_backward_entry(
                    E=self.E,
                    tensors=tensor_ptrs,
                    shapes=self.shapes,
                    gradients=grad_ptrs,
                    final_output_id="loss",
                    batch_size=batch_size_actual
                )

                # ✅ Optimizer 적용
                for name in list(self.weights.keys()) + list(self.biases.keys()):
                    param = self.tensor_map[name]
                    grad_ptr = grad_map.get(name, 0)
                    if grad_ptr == 0:
                        continue

                    grad = cp.ndarray(param.shape, dtype=cp.float32,
                                    memptr=cp.cuda.MemoryPointer(
                                        cp.cuda.UnownedMemory(grad_ptr, param.nbytes, None), 0))

                    # 옵티마이저 버퍼 확보
                    if name not in self.opt_buffers:
                        self.opt_buffers[name] = {
                            "velocity": cp.zeros_like(param),
                            "m": cp.zeros_like(param),
                            "v": cp.zeros_like(param)
                        }

                    velocity = self.opt_buffers[name]["velocity"]
                    m = self.opt_buffers[name]["m"]
                    v = self.opt_buffers[name]["v"]

                    # 옵티마이저 타입 enum
                    opt_type_str = self.optimizer_type.lower()
                    if opt_type_str == "sgd":
                        opt_type_enum = ge.OptimizerType.SGD
                    elif opt_type_str == "momentum":
                        opt_type_enum = ge.OptimizerType.MOMENTUM
                    elif opt_type_str == "adam":
                        opt_type_enum = ge.OptimizerType.ADAM
                    else:
                        raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

                    # CUDA Optimizer 실행
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

                    # NaN 체크 로그 (옵션)
                    if cp.isnan(param).any():
                        logger.warning(f"[NaN] 발생: {name} 업데이트 후 NaN 포함")

                    # 업데이트된 파라미터 저장
                    if name in self.weights:
                        self.weights[name] = param
                    elif name in self.biases:
                        self.biases[name] = param

                    # 🔍 업데이트 후 weight 평균 로그
                    if "w" in name:
                        logger.debug(f"[Epoch {epoch+1}] {name} mean: {cp.mean(param):.6f}")


                self.global_step += 1
                logger.info(f"[Batch 완료] 손실: {loss_val:.10f}")

            # Epoch 마무리 weight 로그 (선택적)
            for name, param in self.weights.items():
                logger.debug(f"[Epoch {epoch+1}] {name} 샘플: {param.ravel()[:5]}")


    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.built:
            raise RuntimeError("✅ 모델이 컴파일되지 않았습니다. 먼저 compile()을 호출하세요.")

        x_cp = cp.asarray(x, dtype=cp.float32)
        batch_size = x_cp.shape[0]

        # 1. 입력 텐서 포인터 준비
        tensor_ptrs = {"input": x_cp.data.ptr}
        self.tensor_map = {"input": x_cp}

        # 2. 가중치 & 편향 준비
        for name, arr in self.weights.items():
            cp_arr = cp.asarray(arr, dtype=cp.float32)
            tensor_ptrs[name] = cp_arr.data.ptr
            self.tensor_map[name] = cp_arr

        for name, arr in self.biases.items():
            cp_arr = cp.asarray(arr, dtype=cp.float32)
            tensor_ptrs[name] = cp_arr.data.ptr
            self.tensor_map[name] = cp_arr

        # 3. 나머지 중간 텐서들 초기화
        for var, shape in self.shapes.items():
            if var not in tensor_ptrs:
                buf = cp.empty((shape.rows, shape.cols), dtype=cp.float32)
                tensor_ptrs[var] = buf.data.ptr
                self.tensor_map[var] = buf

        # 4. 출력 shape 확인 및 초기화
        out_shape = self.shapes[self.output_var]
        
        output_host = np.zeros((batch_size, out_shape.cols), dtype=np.float32)


        # 5. CUDA forward 실행
        ge.run_graph_forward_entry(
            E=self.E,
            tensors=tensor_ptrs,
            shapes=self.shapes,
            out_host=output_host,
            final_output_id=self.output_var,
            batch_size=batch_size
        )

        # 6. 결과 반환 (CPU ndarray)
        return output_host

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        if not self.built:
            raise RuntimeError("✅ 모델이 컴파일되지 않았습니다. 먼저 compile()을 호출하세요.")

        x_cp = cp.asarray(x, dtype=cp.float32)
        y_cp = cp.asarray(y, dtype=cp.float32)
        batch_size = x_cp.shape[0]

        # 1. 입력 및 정답 텐서 포인터
        tensor_ptrs = {
            "input": x_cp.data.ptr,
            "y_true": y_cp.data.ptr
        }
        self.tensor_map = {
            "input": x_cp,
            "y_true": y_cp
        }

        # 2. 가중치 및 편향 포인터
        for name, arr in self.weights.items():
            cp_arr = cp.asarray(arr, dtype=cp.float32)
            tensor_ptrs[name] = cp_arr.data.ptr
            self.tensor_map[name] = cp_arr

        for name, arr in self.biases.items():
            cp_arr = cp.asarray(arr, dtype=cp.float32)
            tensor_ptrs[name] = cp_arr.data.ptr
            self.tensor_map[name] = cp_arr

        # 3. 중간 텐서 버퍼 준비
        for var, shape in self.shapes.items():
            if var not in tensor_ptrs:
                buf = cp.empty((shape.rows, shape.cols), dtype=cp.float32)
                tensor_ptrs[var] = buf.data.ptr
                self.tensor_map[var] = buf

        # 손실 계산 전
        if "loss" in self.tensor_map:
            loss_check = self.tensor_map["loss"]
            cp.cuda.runtime.deviceSynchronize()
            logger.debug(f"[Before Loss] loss buffer: {cp.asnumpy(loss_check.ravel()[:4])}")

        # 4. 손실 계산
        loss_val = ge.run_graph_with_loss_entry(
            E=self.E,
            tensors=tensor_ptrs,
            shapes=self.shapes,
            final_output_id=self.output_var,
            label_tensor_id="y_true",
            loss_type=self.loss_type,
            batch_size=batch_size
        )

        # 손실 계산 후
        if "loss" in self.tensor_map:
            loss_check = self.tensor_map["loss"]
            cp.cuda.runtime.deviceSynchronize()
            logger.debug(f"[After Loss] loss buffer: {cp.asnumpy(loss_check.ravel()[:4])}")


        # 5. 출력 및 메트릭 계산
        output_arr = self.tensor_map[self.output_var]
        y_true_arr = self.tensor_map["y_true"]

        if self.metric_type.lower() == "mse":
            metric_result = metrics.mse(output_arr, y_true_arr)
        elif self.metric_type.lower() == "mae":
            metric_result = metrics.mae(output_arr, y_true_arr)
        elif self.metric_type.lower() == "accuracy":
            metric_result = metrics.accuracy(output_arr, y_true_arr)
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")

        logger.info(f"📊 평가 손실: {loss_val:.10f}, 메트릭({self.metric_type}): {metric_result:.6f}")
        return float(metric_result)
