import numpy as np

OP_TYPES = {
    "sigmoid": 3,
    "relu": 4,
    "tanh": 5,
}

class ActivationMat:
    def __init__(self, activation_name):
        self.activation_name = activation_name
        if activation_name not in OP_TYPES:
            raise ValueError(f"Unsupported activation: {activation_name}")

    def generate_sparse_matrix_block(self, input_ids, node_offset):
        input_len = len(input_ids)
        N = node_offset + input_len

        Conn = np.zeros((N, N), dtype=np.int8)
        OpType = np.zeros((N,), dtype=np.int32)
        ParamIndex = np.full((N,), -1, dtype=np.int32)
        ParamValues = []

        output_ids = []
        for i in range(input_len):
            input_id = input_ids[i]
            nid = node_offset + i
            Conn[input_id, nid] = 1
            OpType[nid] = OP_TYPES[f"{self.activation_name}"]
            output_ids.append(nid)

        return {
            "Conn": Conn,
            "OpType": OpType,
            "ParamIndex": ParamIndex,
            "ParamValues": ParamValues,
            "input_ids": input_ids,
            "output_ids": output_ids,
            "next_node_offset": node_offset + input_len
        }
