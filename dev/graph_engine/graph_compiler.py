import numpy as np
import cupy as cp

class GraphCompiler:
    def __init__(self):
        self.E_rows = []           # [from_idx, to_idx, op_type, W_idx, b_idx]
        self.W_list = []           # 실제 weight
        self.b_list = []           # 실제 bias
        self.W_shapes = []         # backend용 weight shape
        self.b_shapes = []         # backend용 bias shape
        self.node_counter = 0      # auto node 번호 증가용

    def add_layer(self, layer):
        plan = layer.forward_matrix()

        # ✅ input/output index 자동 할당
        from_idx = plan.get("input_idx")
        if from_idx is None:
            from_idx = self.node_counter
        to_idx = plan.get("output_idx")
        if to_idx is None:
            to_idx = from_idx + 1

        op_type = plan["op_type"]

        # ✅ 파라미터 index 처리
        W_idx = -1
        b_idx = -1

        # Python 초기화
        if "W" in plan and plan["W"] is not None:
            W_idx = len(self.W_list)
            self.W_list.append(plan["W"])
        elif "W_shape" in plan:
            W_idx = len(self.W_shapes)
            self.W_shapes.append(plan["W_shape"])

        if "b" in plan and plan["b"] is not None:
            b_idx = len(self.b_list)
            self.b_list.append(plan["b"])
        elif "b_shape" in plan:
            b_idx = len(self.b_shapes)
            self.b_shapes.append(plan["b_shape"])

        self.E_rows.append([from_idx, to_idx, op_type, W_idx, b_idx])

        # ✅ layer 내부에도 node 번호 반영
        layer.input_idx = from_idx
        layer.output_idx = to_idx

        self.node_counter = max(self.node_counter, to_idx + 1)

    def compile_plan(self, use_backend_init=True):
        E = np.array(self.E_rows, dtype=np.int32)
        output_node = self.E_rows[-1][1] if self.E_rows else None

        if use_backend_init:
            return {
                "E": E,
                "W_shapes": self.W_shapes,
                "b_shapes": self.b_shapes,
                "input_node": 0,
                "output_node": output_node
            }
        else:
            return {
                "E": E,
                "W_list": self.W_list,
                "b_list": self.b_list,
                "input_node": 0,
                "output_node": output_node
            }
        
    def prepare_cuda_inputs(compiled):
        # CuPy 배열로 변환
        E_gpu = cp.asarray(compiled["E"], dtype=cp.int32)

        # shape 정보는 그냥 전달하거나 GPU 메모리 공간 할당 시 사용
        W_shapes = compiled.get("W_shapes", [])
        b_shapes = compiled.get("b_shapes", [])

        return {
            "E_gpu": E_gpu,
            "W_shapes": W_shapes,
            "b_shapes": b_shapes,
            "input_node": compiled["input_node"],
            "output_node": compiled["output_node"]
        }

