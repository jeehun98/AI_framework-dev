from __future__ import annotations
from ..ir.nodes import Graph, Op, Tensor

# [MATMUL] → [BIAS_ADD] → [RELU|GELU] 를 하나로 합치는 아주 단순한 데모
def fuse_elementwise(graph: Graph) -> Graph:
    ops = graph.ops
    out: list[Op] = []
    i = 0
    while i < len(ops):
        if (
            i + 2 < len(ops)
            and ops[i].op_type == "MATMUL"
            and ops[i+1].op_type == "BIAS_ADD"
            and ops[i+2].op_type in ("RELU", "GELU")
            and ops[i].outputs[0] is ops[i+1].inputs[0]
            and ops[i+1].outputs[0] is ops[i+2].inputs[0]
        ):
            mm, ba, act = ops[i], ops[i+1], ops[i+2]
            fused_out = Tensor(
                name=act.outputs[0].name,
                shape=act.outputs[0].shape,
                dtype=act.outputs[0].dtype,
                layout=act.outputs[0].layout,
                device=act.outputs[0].device
            )
            fused = Op(
                op_type="GEMM_BIAS_ACT",
                inputs=[mm.inputs[0], mm.inputs[1], ba.inputs[1]],
                outputs=[fused_out],
                attrs={
                    "act": act.op_type,
                    "mnk": mm.attrs.get("mnk", None),   # 있으면 그대로 전달
                },
            )
            out.append(fused)
            i += 3
        else:
            out.append(ops[i])
            i += 1
    return Graph(out)
