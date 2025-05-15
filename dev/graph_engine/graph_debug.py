def print_graph_connections(Conn, OpType, OpTypeLabels=None):
    if OpTypeLabels is None:
        OpTypeLabels = {
            0: "const",
            1: "multiply",
            2: "add",
            3: "sigmoid",
            4: "relu",
            5: "tanh"
        }

    N = Conn.shape[0]
    print("📊 전체 계산 그래프 노드 연결 상태\n")

    for i in range(N):
        # 자신이 어떤 연산자인지
        op = OpType[i]
        op_label = OpTypeLabels.get(op, f"op_{op}")

        # 나를 입력으로 받는 노드들
        outputs = np.where(Conn[i] == 1)[0].tolist()
        # 내가 입력으로 받는 노드들
        inputs = np.where(Conn[:, i] == 1)[0].tolist()

        print(f"🔹 Node {i} ({op_label})")
        if inputs:
            print(f"    🔸 Inputs from: {inputs}")
        if outputs:
            print(f"    🔸 Outputs to : {outputs}")
        if not inputs and not outputs:
            print(f"    ⚠️ Isolated node")
        print()
