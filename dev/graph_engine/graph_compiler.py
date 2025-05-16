# dev/graph_engine/graph_compiler.py
import numpy as np

class GraphCompiler:
    def __init__(self):
        self.Conn = None
        self.OpType = None
        self.ParamIndex = None
        self.ParamValues = []
        self.node_offset = 0
        self.output_ids = []
        self.optype_node_map = {}  # ✅ OpType별 노드 리스트 저장

    def add_layer(self, layer):
        print("\n🧱 [GraphCompiler] Adding layer:", layer.__class__.__name__)
        print("   ↪ input_ids:", self.output_ids or [0, 1, 2, 3])
        print("   ↪ node_offset:", self.node_offset)

        block = layer.generate_sparse_matrix_block(
            input_ids=self.output_ids or [0, 1, 2, 3],
            node_offset=self.node_offset
        )

        Conn_block = block["Conn"]
        OpType_block = block["OpType"]
        ParamIndex_block = block["ParamIndex"]
        ParamValues_block = block["ParamValues"]

        print("   ↪ output_ids:", block["output_ids"])
        print("   ↪ added nodes:", block["next_node_offset"] - self.node_offset)

        N_block = Conn_block.shape[0]
        start = self.node_offset
        end = block["next_node_offset"]

        # 처음 초기화
        if self.Conn is None:
            self.Conn = Conn_block
            self.OpType = OpType_block
            self.ParamIndex = ParamIndex_block
        else:
            N_total = max(self.Conn.shape[0], N_block)
            Conn_new = np.zeros((N_total, N_total), dtype=np.int8)
            Conn_new[:self.Conn.shape[0], :self.Conn.shape[1]] = self.Conn
            Conn_new[:N_block, :N_block] += Conn_block[:N_block, :N_block]
            self.Conn = Conn_new

            OpType_new = np.zeros((N_total,), dtype=np.int32)
            OpType_new[:self.OpType.shape[0]] = self.OpType
            OpType_new[start:end] = OpType_block[start:end]
            self.OpType = OpType_new

            ParamIndex_new = np.full((N_total,), -1, dtype=np.int32)
            ParamIndex_new[:self.ParamIndex.shape[0]] = self.ParamIndex
            ParamIndex_new[start:end] = ParamIndex_block[start:end]
            self.ParamIndex = ParamIndex_new

        self.ParamValues += ParamValues_block
        self.output_ids = block["output_ids"]
        self.node_offset = block["next_node_offset"]

        # ✅ OpType별 노드 ID 정리
        for i in range(start, end):
            op = OpType_block[i]
            if op not in self.optype_node_map:
                self.optype_node_map[op] = []
            self.optype_node_map[op].append(i)

        print("   ↪ updated node_offset:", self.node_offset)

    def get_graph(self):
        return {
            "Conn": self.Conn,
            "OpType": self.OpType,
            "ParamIndex": self.ParamIndex,
            "ParamValues": self.ParamValues,
            "OutputIDs": self.output_ids,
            "TotalNodes": self.node_offset,
            "OpTypeNodeMap": self.optype_node_map  # ✅ 연산자별 노드 분해 결과 포함
        }
