# sequential.py (revised, with y_true shape fix)

import os
import sys
import logging
import numpy as np

# ---- CUDA DLL path first (Windows) ----
try:
    os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
except Exception:
    pass

import cupy as cp

# project-local test helpers if needed
sys.path.append("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor/test")

from dev.layers.layer import Layer
import graph_executor as ge
OpStruct = ge.OpStruct
Shape = ge.Shape

# ---- logging ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_ID = "input"


class Sequential:
    """
    Minimal training loop on custom CUDA graph executor.
    Layers must implement: build(), compute_output_shape(), to_e_matrix()
    """
    def __init__(self, layers=None, trainable=True, name=None, input_shape=None):
        self.built = False
        self._layers = []
        self.input_shape = input_shape
        self.trainable = trainable
        self.name = name
        self.output_var = None

        # device state
        self._device_ready = False
        self.opt_buffers = {}          # {name: {"velocity": cp, "m": cp, "v": cp}}

        # training config
        self.loss_type = None
        self.optimizer_type = None
        self.learning_rate = None
        self.metric_type = None

        # graph and tensors
        self.E_raw = []
        self.E = []
        self.shapes = {}
        self.weights = {}
        self.biases = {}

        # ids
        self.loss_output_id = None

        # step
        self.global_step = 1

        # debug flags
        self.debug_sync = False  # set True to cudaDeviceSynchronize() after kernels

        if layers:
            for layer in layers:
                self.add(layer)

    # ---------- graph building ----------
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
        # reset containers
        self.E_raw = []
        self.weights = {}
        self.biases = {}
        self.metric_type = p_metrics
        self.shapes = {}
        self.opt_buffers = {}
        self._device_ready = False

        input_id = INPUT_ID
        current_shape = self.input_shape

        # build graph from layers
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

        # model final pre-loss output id
        self.output_var = input_id

        # training config
        self.loss_type = loss
        self.optimizer_type = optimizer
        self.learning_rate = learning_rate

        # append loss node
        extra = ge.OpExtraParams()
        extra.label_id = "y_true"
        extra.loss_type = self.loss_type
        self.E_raw.append({
            "op_type": ge.OpType.LOSS,
            "input_id": self.output_var,
            "param_id": "y_true",
            "output_id": "loss",
            "extra_params": extra
        })
        self.loss_output_id = "loss"

        # freeze graph as OpStruct[]
        self.E = []
        for op in self.E_raw:
            extra = op.get("extra_params", ge.OpExtraParams())
            param_id = op.get("param_id", "") or ""
            node = ge.OpStruct(
                ge.OpType(op["op_type"]),
                str(op["input_id"]),
                str(param_id),
                str(op["output_id"]),
                extra
            )
            self.E.append(node)

        self._assert_contiguous_params()
        self.built = True

    # ---------- utils ----------
    def _assert_contiguous_params(self):
        for name, arr in {**self.weights, **self.biases}.items():
            assert hasattr(arr, "flags") and arr.flags.c_contiguous, f"{name} not C-contiguous"

    def _ensure_device_state(self):
        if self._device_ready:
            return
        for name in list(self.weights.keys()) + list(self.biases.keys()):
            param = self.weights.get(name) if name in self.weights else self.biases[name]
            self.opt_buffers[name] = {
                "velocity": cp.zeros_like(param),
                "m": cp.zeros_like(param),
                "v": cp.zeros_like(param)
            }
        self._device_ready = True

    def _opt_type_enum(self):
        s = (self.optimizer_type or "sgd").lower()
        if s == "sgd": return ge.OptimizerType.SGD
        if s == "momentum": return ge.OptimizerType.MOMENTUM
        if s == "adam": return ge.OptimizerType.ADAM
        raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

    def _ensure_label_shape(self, y_cp):
        # Prefer aligning to model output shape.
        if "y_true" in self.shapes:
            return
        out_shape = self.shapes.get(self.output_var, None)
        if out_shape is not None:
            self.shapes["y_true"] = ge.Shape(int(out_shape.rows), int(out_shape.cols))
            return
        C = int(y_cp.shape[-1]) if y_cp.ndim >= 2 else 1
        self.shapes["y_true"] = ge.Shape(1, C)

    # ---------- training ----------
    def train_on_batch(self, x, y):
        if not self.built:
            raise RuntimeError("✅ 모델이 컴파일되지 않았습니다. 먼저 compile()을 호출하세요.")
        self._ensure_device_state()

        x_cp = x if isinstance(x, cp.ndarray) else cp.asarray(x, dtype=cp.float32)
        y_cp = y if isinstance(y, cp.ndarray) else cp.asarray(y, dtype=cp.float32)
        batch_size_actual = x_cp.shape[0]

        # ensure label shape present
        self._ensure_label_shape(y_cp)

        tensor_ptrs = {"input": x_cp.data.ptr, "y_true": y_cp.data.ptr}
        for name, arr in self.weights.items(): tensor_ptrs[name] = arr.data.ptr
        for name, arr in self.biases.items():  tensor_ptrs[name] = arr.data.ptr

        velocity_ptrs = {n: buf["velocity"].data.ptr for n, buf in self.opt_buffers.items()}
        m_ptrs        = {n: buf["m"].data.ptr        for n, buf in self.opt_buffers.items()}
        v_ptrs        = {n: buf["v"].data.ptr        for n, buf in self.opt_buffers.items()}

        loss_val = ge.train_step_entry(
            E=self.E,
            tensors=tensor_ptrs,
            shapes=self.shapes,
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
        if self.debug_sync:
            cp.cuda.runtime.deviceSynchronize()

        self.global_step += 1
        return float(loss_val)

    def fit(self, x=None, y=None, epochs=1, batch_size=-1, verbose=0):
        if not self.built:
            raise RuntimeError("✅ 모델이 컴파일되지 않았습니다. 먼저 compile()을 호출하세요.")

        n = x.shape[0]
        if batch_size == -1 or batch_size > n:
            batch_size = n
        batch_size = max(1, int(batch_size))

        self._ensure_device_state()

        x_full = cp.asarray(x, dtype=cp.float32)
        y_full = cp.asarray(y, dtype=cp.float32)

        # ensure label shape at least once
        self._ensure_label_shape(y_full)

        for epoch in range(epochs):
            idx = np.random.permutation(n)

            epoch_loss = 0.0
            batch_count = 0

            for s in range(0, n, batch_size):
                be = min(s + batch_size, n)
                batch_idx = cp.asarray(idx[s:be], dtype=cp.int32)
                loss_val = self.train_on_batch(x_full[batch_idx], y_full[batch_idx])
                epoch_loss += float(loss_val)
                batch_count += 1

            if verbose and batch_count > 0:
                avg_loss = epoch_loss / batch_count
                logger.info(f"[Epoch {epoch + 1}] 평균 손실: {avg_loss:.6f}")

    # ---------- inference ----------
    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.built:
            raise RuntimeError("✅ 모델이 컴파일되지 않았습니다. 먼저 compile()을 호출하세요.")
        self._ensure_device_state()

        x_cp = cp.asarray(x, dtype=cp.float32)
        batch_size = x_cp.shape[0]

        tensor_ptrs = {"input": x_cp.data.ptr}
        for name, arr in self.weights.items(): tensor_ptrs[name] = arr.data.ptr
        for name, arr in self.biases.items():  tensor_ptrs[name] = arr.data.ptr

        out_shape = self.shapes[self.output_var]
        out_elems = int(out_shape.rows * out_shape.cols)
        output_host = np.zeros((batch_size, out_elems), dtype=np.float32)

        ge.run_graph_forward_entry(
            E=self.E,
            tensors=tensor_ptrs,
            shapes=self.shapes,
            out_host=output_host,
            final_output_id=self.output_var,
            batch_size=batch_size
        )
        if self.debug_sync:
            cp.cuda.runtime.deviceSynchronize()
        return output_host

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        if not self.built:
            raise RuntimeError("✅ 모델이 컴파일되지 않았습니다. 먼저 compile()을 호출하세요.")
        self._ensure_device_state()

        x_cp = cp.asarray(x, dtype=cp.float32)
        y_cp = cp.asarray(y, dtype=cp.float32)
        batch_size = x_cp.shape[0]

        # ensure label shape
        self._ensure_label_shape(y_cp)

        tensor_ptrs = {"input": x_cp.data.ptr, "y_true": y_cp.data.ptr}
        for name, arr in self.weights.items(): tensor_ptrs[name] = arr.data.ptr
        for name, arr in self.biases.items():  tensor_ptrs[name] = arr.data.ptr

        loss_val = ge.run_graph_with_loss_entry(
            E=self.E,
            tensors=tensor_ptrs,
            shapes=self.shapes,
            final_output_id=(self.loss_output_id or self.output_var),
            label_tensor_id="y_true",
            loss_type=self.loss_type,
            batch_size=batch_size
        )
        if self.debug_sync:
            cp.cuda.runtime.deviceSynchronize()
        return float(loss_val)

    # ---------- checkpoints ----------
    def state_dict(self):
        return {
            **{f"W:{k}": cp.asnumpy(v) for k, v in self.weights.items()},
            **{f"b:{k}": cp.asnumpy(v) for k, v in self.biases.items()},
        }

    def load_state_dict(self, sd: dict):
        for k, v in sd.items():
            if k.startswith("W:"):
                name = k[2:]
                if name in self.weights:
                    self.weights[name][...] = cp.asarray(v, dtype=cp.float32)
            elif k.startswith("b:"):
                name = k[2:]
                if name in self.biases:
                    self.biases[name][...] = cp.asarray(v, dtype=cp.float32)
        self._device_ready = False
        self._ensure_device_state()
