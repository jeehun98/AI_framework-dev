# dev/models/sequential.py
# (revised: y_true shape fix + vector/legacy-compatible OpStruct freeze)

import os
import sys
import logging
import numpy as np

try:
    os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
except Exception:
    pass

import cupy as cp

sys.path.append("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor/test")

from dev.layers.layer import Layer
import graph_executor as ge
OpStruct = ge.OpStruct
Shape = ge.Shape

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_ID = "input"


class Sequential:
    def __init__(self, layers=None, trainable=True, name=None, input_shape=None):
        self.built = False
        self._layers = []
        self.input_shape = input_shape
        self.trainable = trainable
        self.name = name
        self.output_var = None

        self._device_ready = False
        self.opt_buffers = {}

        self.loss_type = None
        self.optimizer_type = None
        self.learning_rate = None
        self.metric_type = None

        self.E_raw = []     # list[dict | ge.OpStruct]
        self.E = []         # list[ge.OpStruct]
        self.shapes = {}
        self.weights = {}
        self.biases = {}

        self.loss_output_id = None
        self.global_step = 1
        self.debug_sync = False

        if layers:
            for layer in layers:
                self.add(layer)

    # ---------- graph building ----------
    def add(self, layer):
        if not isinstance(layer, Layer):
            raise ValueError("Only instances of Layer can be added.")

        if not self._layers:
            inferred = layer.input_shape or self.input_shape
            if inferred is None:
                raise RuntimeError("첫 번째 레이어는 input_shape를 지정해야 합니다.")
            layer.input_shape = inferred
            layer.build(inferred)
        else:
            prev = self._layers[-1]
            input_shape = prev.output_shape
            if input_shape is None:
                raise RuntimeError("이전 레이어가 build되지 않았습니다.")
            layer.input_shape = input_shape
            layer.build(input_shape)

        self._layers.append(layer)
        logger.info(f"✅ 레이어 추가됨: {layer.__class__.__name__} (input_shape={layer.input_shape}, output_shape={layer.output_shape})")

    # ---------- helpers ----------
    def _to_opstruct(self, node):
        """
        dict 또는 ge.OpStruct -> ge.OpStruct
        벡터 API가 오더라도, 레거시 필드(input_id/param_id)도 반드시 채워서
        레거시 경로(LOSS 등)와 완전 호환되도록 만든다.
        """
        if isinstance(node, ge.OpStruct):
            # 파이프라인 어딘가에서 legacy를 읽을 수 있으니 normalize 호출을 기대
            return node

        op_type = node["op_type"]
        if isinstance(op_type, int):
            op_type = ge.OpType(op_type)

        out_id = node["output_id"]
        extra  = node.get("extra_params", ge.OpExtraParams())

        inputs = node.get("inputs", [])
        params = node.get("params", [])

        # legacy 보정
        if not inputs and node.get("input_id", ""):
            inputs = [node["input_id"]]
        if not params and node.get("param_id", ""):
            params = [node["param_id"]]

        # 안전 보정
        if isinstance(inputs, str): inputs = [inputs]
        if isinstance(params, str): params = [params]

        # ✅ 레거시 필드도 반드시 채운다 (LOSS 등 호환)
        legacy_in    = inputs[0] if inputs else node.get("input_id", "")
        legacy_param = params[0] if params else node.get("param_id", "")

        # 레거시 생성자 사용 (pybind 레거시 경로와 100% 호환)
        op = ge.OpStruct(op_type, str(legacy_in), str(legacy_param), str(out_id), extra)

        # 가능하면 벡터 필드도 채워주기 (pybind가 쓰기 가능할 때)
        try:
            if hasattr(op, "inputs") and not getattr(op, "inputs"):
                if inputs: op.inputs = inputs
            if hasattr(op, "params") and not getattr(op, "params"):
                if params: op.params = params
        except Exception:
            # 쓰기 불가한 바인딩이면 레거시만으로도 동작함
            pass

        return op

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
        if "y_true" in self.shapes:
            return
        out_shape = self.shapes.get(self.output_var, None)
        if out_shape is not None:
            self.shapes["y_true"] = ge.Shape(int(out_shape.rows), int(out_shape.cols))
            return
        C = int(y_cp.shape[-1]) if y_cp.ndim >= 2 else 1
        self.shapes["y_true"] = ge.Shape(1, C)

    # ---------- compile / graph build ----------
    def compile(self, optimizer='sgd', loss='mse', p_metrics='mse', learning_rate=0.001):
        self.E_raw = []
        self.weights = {}
        self.biases = {}
        self.metric_type = p_metrics
        self.shapes = {}
        self.opt_buffers = {}
        self._device_ready = False

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
            if w: self.weights.update(w)
            if b: self.biases.update(b)
            if shape_map:
                for k, shp in shape_map.items():
                    if isinstance(shp, ge.Shape):
                        self.shapes[k] = shp
                    else:
                        self.shapes[k] = ge.Shape(int(shp[0]), int(shp[1]))
            input_id = output_id

        self.output_var = input_id
        self.loss_type = loss
        self.optimizer_type = optimizer
        self.learning_rate = learning_rate

        # ✅ LOSS 노드 추가 — 벡터든 레거시든 OK지만, 레거시 필드는 반드시 채우자.
        loss_extra = ge.OpExtraParams()
        loss_extra.label_id = "y_true"
        loss_extra.loss_type = self.loss_type
        self.E_raw.append({
            "op_type": ge.OpType.LOSS,
            "inputs":  [self.output_var],   # vector
            "params":  ["y_true"],          # vector
            "input_id": self.output_var,    # legacy 채우기
            "param_id": "y_true",           # legacy 채우기
            "output_id": "loss",
            "extra_params": loss_extra
        })
        self.loss_output_id = "loss"

        # freeze
        self.E = []
        for node in self.E_raw:
            self.E.append(self._to_opstruct(node))

        self._assert_contiguous_params()
        self.built = True

    # ---------- training ----------
    def train_on_batch(self, x, y):
        if not self.built:
            raise RuntimeError("✅ 모델이 컴파일되지 않았습니다. 먼저 compile()을 호출하세요.")
        self._ensure_device_state()

        x_cp = x if isinstance(x, cp.ndarray) else cp.asarray(x, dtype=cp.float32)
        y_cp = y if isinstance(y, cp.ndarray) else cp.asarray(y, dtype=cp.float32)
        batch_size_actual = x_cp.shape[0]

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

    def grad_dump_l2(self, x_batch, y_batch, keys=None, head=8, title="Grad L2 dump"):
        if not self.built:
            raise RuntimeError("compile() 먼저 호출하세요.")
        self._ensure_device_state()

        x_cp = cp.asarray(x_batch, dtype=cp.float32)
        y_cp = cp.asarray(y_batch, dtype=cp.float32)

        self._ensure_label_shape(y_cp)

        tensor_ptrs = {"input": x_cp.data.ptr, "y_true": y_cp.data.ptr}
        for name, arr in self.weights.items(): tensor_ptrs[name] = arr.data.ptr
        for name, arr in self.biases.items():  tensor_ptrs[name] = arr.data.ptr

        grads_ptrs = ge.run_graph_backward_entry(
            E=self.E,
            tensors=tensor_ptrs,
            shapes=self.shapes,
            gradients={},                     # C++ 쪽에서 채워 반환
            final_output_id=self.output_var,  # 손실 이전 출력
            batch_size=x_cp.shape[0]
        )

        if keys is None:
            keys = list(self.weights.keys()) + list(self.biases.keys())

        print(f"\n=== {title} (batch={x_cp.shape[0]}) ===")
        found_any = False
        for k in keys:
            if k not in grads_ptrs:
                print(f"[MISS] {k}: grad not found")
                continue
            shp = self.shapes.get(k, None)
            if shp is None:
                print(f"[MISS] {k}: shape not found")
                continue

            ptr = int(grads_ptrs[k])
            nbytes = int(shp.rows * shp.cols) * 4

            mem = cp.cuda.UnownedMemory(ptr, nbytes, self)
            mp  = cp.cuda.MemoryPointer(mem, 0)
            gcp = cp.ndarray((int(shp.rows), int(shp.cols)), dtype=cp.float32, memptr=mp)

            g_np = cp.asnumpy(gcp)
            l2 = float(np.linalg.norm(g_np))
            flat = g_np.reshape(-1)
            head_vals = " ".join([f"{v:+.3e}" for v in flat[:head]])
            print(f"[GRAD] {k:>32s} | shape=({shp.rows},{shp.cols}) | L2={l2:.6e} | head: {head_vals}")
            found_any = True

        if not found_any:
            print("❌ No gradients returned. (BWD 경로/노드 연결/손실 설정을 점검하세요.)")
