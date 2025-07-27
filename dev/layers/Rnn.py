import sys
sys.path.append("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor")
import graph_executor as ge  # Pybind11 모듈

# graph_executor에서 정의된 Shape 사용

import cupy as cp
from dev.layers.layer import Layer

class RNN(Layer):
    def __init__(self, units, activation=cp.tanh, input_shape=None, name=None, initializer='xavier', use_backend_init=False, **kwargs):
        super().__init__(name, **kwargs)
        self.units = units
        self.activation = activation
        self.initializer = initializer
        self.use_backend_init = use_backend_init

        self.input_shape = input_shape
        self.output_shape = None

        self.Wx = None
        self.Wh = None
        self.b = None

        self.last_inputs = None
        self.last_hs = None

        self.dWx = None
        self.dWh = None
        self.db = None

        # ✅ GraphCompiler용 변수
        self.name = name or f"rnn_{id(self)}"
        self.input_idx = None
        self.output_idx = None

    def build(self, input_shape):
        if len(input_shape) == 2:
            time_steps, input_dim = input_shape
        elif len(input_shape) == 3:
            _, time_steps, input_dim = input_shape
        else:
            raise ValueError(f"RNN input_shape 오류: {input_shape}")

        self.output_shape = (None, self.units)

        if self.use_backend_init:
            # shape 정보만 저장하고 backend 초기화로 대체
            return

        def init(shape):
            if self.initializer == 'xavier':
                stddev = cp.sqrt(1. / shape[0])
                return cp.random.randn(*shape).astype(cp.float32) * stddev
            elif self.initializer == 'he':
                stddev = cp.sqrt(2. / shape[0])
                return cp.random.randn(*shape).astype(cp.float32) * stddev
            elif self.initializer == 'zeros':
                return cp.zeros(shape, dtype=cp.float32)
            elif self.initializer == 'ones':
                return cp.ones(shape, dtype=cp.float32)
            else:
                raise ValueError(f"Unknown initializer: {self.initializer}")

        self.Wx = init((input_dim, self.units))
        self.Wh = init((self.units, self.units))
        self.b = cp.zeros((1, self.units), dtype=cp.float32)

    def call(self, input_seq):
        input_seq = cp.asarray(input_seq, dtype=cp.float32)
        B, T, D = input_seq.shape
        self.input_shape = input_seq.shape
        self.output_shape = (B, self.units)

        # 출력 버퍼 준비
        output = cp.zeros((B, self.units), dtype=cp.float32)

        # 포인터 설정
        x_ptr = input_seq.data.ptr
        Wx_ptr = self.Wx.data.ptr if self.Wx is not None else 0
        Wh_ptr = self.Wh.data.ptr if self.Wh is not None else 0
        b_ptr = self.b.data.ptr if self.b is not None else 0
        out_ptr = output.data.ptr

        # Shape 정보
        shapes = {
            "x": ge.Shape(B, T * D),
            "Wx": ge.Shape(D, self.units),
            "Wh": ge.Shape(self.units, self.units),
            "b": ge.Shape(1, self.units),
            "out": ge.Shape(B, self.units)
        }

        # Tensor 포인터 매핑
        tensors = {
            "x": x_ptr,
            "Wx": Wx_ptr,
            "Wh": Wh_ptr,
            "b": b_ptr,
            "out": out_ptr
        }

        # Extra 정보
        extra = ge.OpExtraParams()
        extra.batch_size = B
        extra.input_h = T  # time_steps
        extra.input_w = D  # input_dim
        extra.hidden_size = self.units

        # 연산 노드
        op = ge.OpStruct(ge.OpType.RNN, "x", "Wx", "out", extra)
        op.param2 = "Wh"
        op.param3 = "b"
        E = [op]

        # CUDA 실행
        ge.run_graph_cuda(E, tensors, shapes, out_ptr, "out", B)

        return output

    
    def backward(self, grad_output):
        B, _ = grad_output.shape
        _, T, D = self.input_shape

        # 포인터 설정
        grad_out_ptr = grad_output.data.ptr
        x_ptr = self.last_input.data.ptr
        Wx_ptr = self.Wx.data.ptr
        Wh_ptr = self.Wh.data.ptr
        b_ptr = self.b.data.ptr

        # Shape 정보
        shapes = {
            "x": ge.Shape(B, T * D),
            "Wx": ge.Shape(D, self.units),
            "Wh": ge.Shape(self.units, self.units),
            "b": ge.Shape(1, self.units),
            "out": ge.Shape(B, self.units)
        }

        tensors = {
            "x": x_ptr,
            "Wx": Wx_ptr,
            "Wh": Wh_ptr,
            "b": b_ptr,
            "out": grad_out_ptr
        }

        grad_ptrs = {}

        op = ge.OpStruct(ge.OpType.RNN, "x", "Wx", "out", ge.OpExtraParams())
        op.param2 = "Wh"
        op.param3 = "b"
        E = [op]

        grads = ge.run_graph_backward(E, tensors, shapes, grad_ptrs, "out", B)

        # 역전파 결과 추출
        self.dWx = self._copy_from_ptr(grads.get("Wx"), (D, self.units))
        self.dWh = self._copy_from_ptr(grads.get("Wh"), (self.units, self.units))
        self.db = self._copy_from_ptr(grads.get("b"), (1, self.units))
        dx = self._copy_from_ptr(grads.get("x"), (B, T * D)).reshape((B, T, D))

        return dx

    def update(self, optimizer):
        optimizer.update(self.Wx, self.dWx, self.Wh, self.dWh, self.b, self.db)

    # ✅ GraphCompiler용 연산 정보 정의
    def forward_matrix(self, input_name="input"):
        input_dim = self.input_shape[2] if self.input_shape and len(self.input_shape) == 3 else None

        if self.use_backend_init:
            return {
                "input_idx": self.input_idx,
                "output_idx": self.output_idx,
                "op_type": 10,  # 예: RNN 연산 코드
                "W_shape": (input_dim, self.units),
                "Wh_shape": (self.units, self.units),
                "b_shape": (1, self.units),
                "W": None,
                "Wh": None,
                "b": None
            }
        else:
            return {
                "input_idx": self.input_idx,
                "output_idx": self.output_idx,
                "op_type": 10,
                "W": self.Wx,
                "Wh": self.Wh,
                "b": self.b
            }

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def _copy_from_ptr(self, ptr, shape):
        size = int(cp.prod(cp.array(shape)))
        mem = cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(ptr, size * 4, None), 0)
        return cp.ndarray(shape, dtype=cp.float32, memptr=mem)
