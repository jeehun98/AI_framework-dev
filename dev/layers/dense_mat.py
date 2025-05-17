import numpy as np

OP_TYPES = {
    "const": 0,
    "multiply": 1,
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
            self.weights = (np.random.randn(input_dim, self.output_dim) * stddev).astype(np.float32)
        elif self.initializer == 'xavier':
            stddev = np.sqrt(1. / input_dim)
            self.weights = (np.random.randn(input_dim, self.output_dim) * stddev).astype(np.float32)
        else:
            print(f"[WARN] Unknown initializer '{self.initializer}', using uniform [-1, 1]")
            self.weights = np.random.uniform(-1, 1, size=(input_dim, self.output_dim)).astype(np.float32)

        self.bias = np.zeros((1, self.output_dim), dtype=np.float32)

        # 디버깅용 출력
        print("[DEBUG] weights initialized:", self.weights)

    def generate_sparse_matrix_block(self, input_ids, node_offset):
        input_dim = self.input_dim
        output_dim = self.output_dim
        
        weight_const_count = input_dim * output_dim
        bias_const_count = output_dim
        mul_count = input_dim * output_dim
        add_count = (input_dim - 1 + 1) * output_dim  # add chain + bias add
        total_nodes = weight_const_count + bias_const_count + mul_count + add_count
        N = node_offset + total_nodes

        Conn = np.zeros((N, N), dtype=np.int8)
        OpType = np.zeros((N,), dtype=np.int32)
        ParamIndex = np.full((N,), -1, dtype=np.int32)
        ParamValues = []

        nid = node_offset

        # 1️⃣ Weight constants
        weight_ids = np.zeros((input_dim, output_dim), dtype=np.int32)
        for i in range(input_dim):
            for j in range(output_dim):
                OpType[nid] = OP_TYPES["const"]
                ParamIndex[nid] = len(ParamValues)
                ParamValues.append(self.weights[i, j])
                weight_ids[i, j] = nid
                nid += 1

        # 2️⃣ Bias constants
        bias_ids = np.zeros((output_dim,), dtype=np.int32)
        for j in range(output_dim):
            OpType[nid] = OP_TYPES["const"]
            ParamIndex[nid] = len(ParamValues)
            ParamValues.append(self.bias[0, j])
            bias_ids[j] = nid
            nid += 1

        # 3️⃣ Multiply nodes: x_i * W_ij
        mul_ids = np.zeros((input_dim, output_dim), dtype=np.int32)
        for j in range(output_dim):
            for i in range(input_dim):
                OpType[nid] = OP_TYPES["multiply"]
                Conn[input_ids[i], nid] = 1
                Conn[weight_ids[i, j], nid] = 1
                mul_ids[i, j] = nid
                nid += 1

        # 4️⃣ Add chains: sum(x_i * W_ij) + b_j
        output_ids = np.zeros((output_dim,), dtype=np.int32)
        for j in range(output_dim):
            # 첫 두 곱셈 항
            prev = nid
            OpType[nid] = OP_TYPES["add"]
            Conn[mul_ids[0, j], nid] = 1
            Conn[mul_ids[1, j], nid] = 1
            nid += 1

            # 나머지 항
            for i in range(2, input_dim):
                OpType[nid] = OP_TYPES["add"]
                Conn[prev, nid] = 1
                Conn[mul_ids[i, j], nid] = 1
                prev = nid
                nid += 1

            # 마지막 bias add
            OpType[nid] = OP_TYPES["add"]
            Conn[prev, nid] = 1
            Conn[bias_ids[j], nid] = 1
            output_ids[j] = nid
            nid += 1

        return {
            "Conn": Conn,
            "OpType": OpType,
            "ParamIndex": ParamIndex,
            "ParamValues": ParamValues,
            "input_ids": input_ids,
            "output_ids": output_ids,
            "weight_ids": weight_ids,
            "bias_ids": bias_ids,
            "mul_ids": mul_ids,
            "next_node_offset": nid
        }
