from collections import OrderedDict

def print_graph(node_list):
    """
    계산 그래프를 트리 형태로 출력 (고유 ID 및 자식 개수 포함)
    """
    def print_node(node, prefix="", is_last=True, visited=None):
        if visited is None:
            visited = OrderedDict()

        if node in visited:
            print(prefix + ("└── " if is_last else "├── ") +
                  f"[{node.operation}] out={node.output} weight={node.weight_value} id={id(node)} (↺ visited)")
            return

        connector = "└── " if is_last else "├── "
        print(prefix + connector +
              f"[{node.operation}] out={node.output} weight={node.weight_value} "
              f"id={id(node)} | children_root_nodes={len(node.children_root_nodes)}")

        visited[node] = True

        child_count = len(node.children_root_nodes)
        for idx, child in enumerate(node.children_root_nodes):
            is_last_child = (idx == child_count - 1)
            next_prefix = prefix + ("    " if is_last else "│   ")
            print_node(child, next_prefix, is_last_child, visited)

    if not node_list:
        print("그래프 비어있음")
        return

    print("\n[ 계산 그래프 구조 ]")
    visited = OrderedDict()
    for idx, node in enumerate(node_list):
        print(f"\n🌱 Root Node {idx} ({node.operation}) id={id(node)}")
        print_node(node, is_last=(idx == len(node_list) - 1), visited=visited)

def connect_graphs(parent_leaf_nodes, children_root_nodes):
    """
    계산 그래프 노드 연결 (n:m 대응)
    
    - 각 parent 노드를 동일한 간격으로 여러 children_root_nodes에 연결
    - 예: parent가 10개, child가 30개면 → 각 parent는 연속된 3개의 child에 연결됨
    """
    print(len(parent_leaf_nodes), len(children_root_nodes), "개수 확인")

    if not parent_leaf_nodes or not children_root_nodes:
        raise ValueError("두 node_list 중 하나 이상이 비어 있습니다.")

    if len(children_root_nodes) % len(parent_leaf_nodes) != 0:
        raise ValueError(f"children_root_nodes 수가 parent_leaf_nodes 수의 배수가 아닙니다. parent_leaf_nodes={len(parent_leaf_nodes)}, children_root_nodes={len(children_root_nodes)}")

    ratio = len(children_root_nodes) // len(parent_leaf_nodes)

    for i, parent in enumerate(parent_leaf_nodes):
        for j in range(ratio):
            child = children_root_nodes[i * ratio + j]
            child.add_child(parent)

    return children_root_nodes
