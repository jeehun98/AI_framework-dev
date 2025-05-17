import numpy as np

OP_TYPES = {
    "const": 0,
    "multiply": 1,
    "add": 2,
    "neg": 3,
    "exp": 4,
    "div": 5,
    "max": 6,
    "subtract": 7,
}


class ActivationMat:
    def __init__(self, activation_name):
        self.activation_name = activation_name

    def build(self, input_dim):
        self.input_dim = input_dim
        self.output_dim = input_dim 

    def generate_sparse_matrix_block(self, input_ids, node_offset):
        Conn = []
        OpType = []
        ParamIndex = []
        ParamValues = []
        output_ids = []

        nid = node_offset  # 절대 노드 ID 시작점

        N_estimate = node_offset + len(input_ids) * 6  # sigmoid = 6노드 구조
        Conn = np.zeros((N_estimate, N_estimate), dtype=np.int8)
        OpType = np.zeros((N_estimate,), dtype=np.int32)
        ParamIndex = np.full((N_estimate,), -1, dtype=np.int32)

        for input_id in input_ids:
            # const 1
            const1_id = nid
            OpType[const1_id] = OP_TYPES["const"]
            ParamIndex[const1_id] = len(ParamValues)
            ParamValues.append(1.0)
            nid += 1

            # neg
            neg_id = nid
            OpType[neg_id] = OP_TYPES["neg"]
            Conn[input_id, neg_id] = 1
            nid += 1

            # exp
            exp_id = nid
            OpType[exp_id] = OP_TYPES["exp"]
            Conn[neg_id, exp_id] = 1
            nid += 1

            # add
            add_id = nid
            OpType[add_id] = OP_TYPES["add"]
            Conn[const1_id, add_id] = 1
            Conn[exp_id, add_id] = 1
            nid += 1

            # div
            div_id = nid
            OpType[div_id] = OP_TYPES["div"]
            Conn[const1_id, div_id] = 1
            Conn[add_id, div_id] = 1
            output_ids.append(div_id)
            nid += 1

        return {
            "Conn": Conn[:nid, :nid],  # 잘라서 반환
            "OpType": OpType[:nid],
            "ParamIndex": ParamIndex[:nid],
            "ParamValues": ParamValues,
            "input_ids": input_ids,
            "output_ids": output_ids,
            "next_node_offset": nid
        }
