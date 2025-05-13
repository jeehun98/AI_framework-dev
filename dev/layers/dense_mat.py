import numpy as np

OP_TYPES = {
    "const": 0,
    "matmul": 1,
    "add": 2,
}

class DenseMat:
    def __init__(self, units, input_dim=None, initializer='he'):
        self.units = units
        self.input_dim = input_dim
        self.output_dim = units
        self.initializer = initializer
        self.weights = None
        self.bias = None

    def build(self, input_dim):
        self.input_dim = input_dim
        self.output_dim = self.units

        if self.initializer == 'he':
            stddev = np.sqrt(2. / input_dim)
            self.weights = (np.random.randn(input_dim, self.units) * stddev).astype(np.float32)
        elif self.initializer == 'xavier':
            stddev = np.sqrt(1. / input_dim)
            self.weights = (np.random.randn(input_dim, self.units) * stddev).astype(np.float32)
        else:
            self.weights = np.zeros((input_dim, self.units), dtype=np.float32)

        self.bias = np.zeros((1, self.units), dtype=np.float32)

    def generate_sparse_matrix_block(self, input_ids, node_offset):
        total_nodes = 4  # const W, const b, matmul, add
        Conn = np.zeros((node_offset + total_nodes, node_offset + total_nodes), dtype=np.int8)
        OpType = np.zeros((node_offset + total_nodes,), dtype=np.int32)
        ParamIndex = np.full((node_offset + total_nodes,), -1, dtype=np.int32)
        ParamValues = []

        nid = node_offset

        w_id = nid
        OpType[w_id] = OP_TYPES["const"]
        ParamIndex[w_id] = len(ParamValues)
        ParamValues.append(self.weights)
        nid += 1

        b_id = nid
        OpType[b_id] = OP_TYPES["const"]
        ParamIndex[b_id] = len(ParamValues)
        ParamValues.append(self.bias)
        nid += 1

        matmul_id = nid
        OpType[matmul_id] = OP_TYPES["matmul"]
        Conn[input_ids[0], matmul_id] = 1
        Conn[w_id, matmul_id] = 1
        nid += 1

        add_id = nid
        OpType[add_id] = OP_TYPES["add"]
        Conn[matmul_id, add_id] = 1
        Conn[b_id, add_id] = 1
        nid += 1

        return {
            "Conn": Conn,
            "OpType": OpType,
            "ParamIndex": ParamIndex,
            "ParamValues": ParamValues,
            "input_ids": input_ids,
            "output_ids": [add_id],
            "next_node_offset": nid
        }
