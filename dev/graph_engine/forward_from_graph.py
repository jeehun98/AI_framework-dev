import numpy as np
from collections import deque

OP_TYPES = {
    0: "const",
    1: "multiply",
    2: "add",
    3: "sigmoid",
    4: "relu",
    5: "tanh",
}

def topological_sort(Conn):
    N = Conn.shape[0]
    indegree = np.sum(Conn, axis=0)
    queue = deque([i for i in range(N) if indegree[i] == 0])

    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in np.where(Conn[node] == 1)[0]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    return order

def forward_from_graph(Conn, OpType, ParamIndex, ParamValues, input_values):
    N = Conn.shape[0]
    Value = [None] * N

    for node_id, val in input_values.items():
        Value[node_id] = val

    sorted_nodes = topological_sort(Conn)

    for i in sorted_nodes:
        if Value[i] is not None:
            continue

        op = OpType[i]

        input_ids = np.where(Conn[:, i] == 1)[0].tolist()
        print(f"[!!] Node {i} OpType={OpType[i]} input_ids={input_ids}")
        input_vals = [Value[idx] for idx in input_ids]


        if op == 0:  # const
            Value[i] = ParamValues[ParamIndex[i]]
        elif op == 1:  # multiply
            print(f"[*] Multiply Node {i}: inputs = {input_vals}")
            Value[i] = input_vals[0] * input_vals[1]

            Value[i] = input_vals[0] * input_vals[1]
        elif op == 2:  # add
            Value[i] = input_vals[0] + input_vals[1]
        elif op == 3:  # sigmoid
            Value[i] = 1 / (1 + np.exp(-input_vals[0]))
            
        elif op == 4:  # relu
            Value[i] = max(0, input_vals[0])
        elif op == 5:  # tanh
            Value[i] = np.tanh(input_vals[0])
        else:
            raise ValueError(f"Unsupported OpType: {op}")

    return Value
