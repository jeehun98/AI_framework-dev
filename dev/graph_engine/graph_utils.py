from collections import OrderedDict


def get_leaf_nodes(node_list):
    """
    자식이 없는 노드들을 리프 노드로 간주하고 반환합니다.
    """

    node_list[0].print_tree()

    if not node_list:
        raise ValueError("node_list가 비어 있습니다.")

    visited = OrderedDict()
    leaf_nodes = []

    def dfs(node):
        if node in visited:
            return
        visited[node] = True
        if not node.children:
            leaf_nodes.append(node)
        else:
            for child in node.children:
                dfs(child)

    for node in node_list:
        dfs(node)

    return leaf_nodes


def get_root_nodes(node_list):
    """
    부모가 없는 노드들을 루트 노드로 간주하고 반환합니다.
    """
    return [node for node in node_list if not node.parents]


def connect_graphs(node_list1, node_list2):
    """
    node_list1의 리프 노드를 node_list2의 루트 노드와 순서대로 연결합니다.
    """
    if not node_list1 or not node_list2:
        raise ValueError("두 node_list 중 하나 이상이 비어 있습니다.")

    leaf_nodes = get_leaf_nodes(node_list1)
    root_nodes = node_list2

    print(len(leaf_nodes), len(root_nodes), "확인용 2222")

    if len(leaf_nodes) != len(root_nodes):
        raise ValueError(f"리프 노드({len(leaf_nodes)})와 루트 노드({len(root_nodes)})의 개수가 일치하지 않습니다.")

    for leaf, root in zip(leaf_nodes, root_nodes):
        leaf.add_child(root)
        root.add_parent(leaf)

    return node_list1


def print_graph(node_list):
    """
    터미널에서 보기 좋은 트리 구조로 계산 그래프를 출력합니다.
    """
    def print_node(node, prefix="", is_last=True, visited=None):
        if visited is None:
            visited = OrderedDict()
        if node in visited:
            print(prefix + ("└── " if is_last else "├── ") + f"[{node.operation}] out={node.output} (↺ visited)")
            return
        visited[node] = True

        connector = "└── " if is_last else "├── "
        print(prefix + connector + f"[{node.operation}] out={node.output} weight={node.weight_value}")

        child_count = len(node.children)
        for idx, child in enumerate(node.children):
            is_last_child = (idx == child_count - 1)
            next_prefix = prefix + ("    " if is_last else "│   ")
            print_node(child, next_prefix, is_last_child, visited)

    if not node_list:
        print("그래프 비어있음")
        return

    print("\n[ 그래프 구조 ]")
    visited = OrderedDict()
    for idx, node in enumerate(node_list):
        print_node(node, is_last=(idx == len(node_list) - 1), visited=visited)
