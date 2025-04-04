# dev/cal_graph/graph_utils.py

def get_leaf_nodes(node_list):
    if not node_list:
        raise ValueError("node_list가 비어 있습니다.")
    visited = set()
    leaf_nodes = []

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        if not node.children:
            leaf_nodes.append(node)
        else:
            for child in node.children:
                dfs(child)

    for node in node_list:
        dfs(node)

    return leaf_nodes


def connect_graphs(node_list1, node_list2):
    if not node_list1:
        raise ValueError("첫 번째 node_list가 비어 있습니다.")
    if not node_list2:
        raise ValueError("두 번째 node_list가 비어 있습니다.")

    leaf_nodes = get_leaf_nodes(node_list1)
    root_nodes = node_list2

    if not leaf_nodes:
        raise ValueError("첫 번째 node_list에서 리프 노드를 찾을 수 없습니다.")

    for i in range(len(leaf_nodes)):
        leaf_nodes[i].add_child(root_nodes[i])
        root_nodes[i].add_parent(leaf_nodes[i])

    return node_list1


def print_graph(node_list):
    def print_node(node, prefix="", is_last=True, visited=None):
        if visited is None:
            visited = set()
        if node in visited:
            return
        visited.add(node)

        connector = "└── " if is_last else "├── "
        print(prefix + connector + f"[{node.operation}] out={node.output}")

        child_count = len(node.children)
        for idx, child in enumerate(node.children):
            is_last_child = (idx == child_count - 1)
            next_prefix = prefix + ("    " if is_last else "│   ")
            print_node(child, next_prefix, is_last_child, visited)

    if not node_list:
        print("그래프 비어있음")
        return

    print("\n[ 그래프 구조 ]")
    for idx, node in enumerate(node_list):
        print_node(node, is_last=(idx == len(node_list) - 1))
