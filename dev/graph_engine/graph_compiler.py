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

        input_ids = self.output_ids.tolist() if isinstance(self.output_ids, np.ndarray) else self.output_ids
        input_ids = input_ids if len(input_ids) > 0 else [0, 1, 2, 3]

        print("   ↪ input_ids:", input_ids)
        print("   ↪ node_offset:", self.node_offset)

        block = layer.generate_sparse_matrix_block(
            input_ids=input_ids,
            node_offset=self.node_offset
        )

        Conn_block = block["Conn"]
        OpType_block = block["OpType"]
        ParamIndex_block = block["ParamIndex"]
        ParamValues_block = block["ParamValues"]

        start = self.node_offset
        end = block["next_node_offset"]
        N_block = end
        print("   ↪ output_ids:", block["output_ids"])
        print("   ↪ added nodes:", end - start)

        # 병합할 전체 크기 계산
        N_total = max(
            self.Conn.shape[0] if self.Conn is not None else 0,
            N_block
        )

        # Conn 병합
        Conn_new = np.zeros((N_total, N_total), dtype=np.int8)
        if self.Conn is not None:
            Conn_new[:self.Conn.shape[0], :self.Conn.shape[1]] = self.Conn
        Conn_new[start:end, start:end] = Conn_block[start:end, start:end]
        self.Conn = Conn_new

        # OpType 병합
        OpType_new = np.zeros((N_total,), dtype=np.int32)
        if self.OpType is not None:
            OpType_new[:self.OpType.shape[0]] = self.OpType
        OpType_new[start:end] = OpType_block[start:end]
        self.OpType = OpType_new

        # ParamIndex 병합
        ParamIndex_new = np.full((N_total,), -1, dtype=np.int32)
        if self.ParamIndex is not None:
            ParamIndex_new[:self.ParamIndex.shape[0]] = self.ParamIndex
        ParamIndex_new[start:end] = ParamIndex_block[start:end]
        self.ParamIndex = ParamIndex_new

        # ParamValues 병합
        self.ParamValues += ParamValues_block

        # 출력 노드 및 오프셋 갱신
        self.output_ids = block["output_ids"]
        self.node_offset = block["next_node_offset"]

        # ✅ OpType별 노드 ID 정리
        for i in range(start, end):
            op = self.OpType[i]
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
