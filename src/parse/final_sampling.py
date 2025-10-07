import pandas as pd
from pathlib import Path


def parse_graph_file(file_path: str):
    """
    Parse a graph file and create DataFrames with edge and node information.

    Args:
        file_path: Path to the graph file

    Returns:
        Tuple of (edges_df, nodes_df, graph_sizes) where:
        - edges_df: DataFrame with columns: src, dst, is_laundering, affinity
        - nodes_df: DataFrame with columns: node_id, label
        - graph_sizes: List of integers representing the size (number of nodes) of each graph
    """
    edges = []
    nodes = []
    graph_sizes = []
    graph_id = 0

    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    i = 0
    while i < len(lines):
        # Skip empty lines
        if not lines[i]:
            i += 1
            continue

        # Parse N
        if lines[i].startswith('N='):
            graph_id += 1
            n = int(lines[i].split('=')[1])
            graph_sizes.append(n)
            i += 1

            # Parse X (node labels)
            if i < len(lines) and lines[i] == 'X:':
                i += 1
                node_labels = list(map(int, lines[i].split()))
                i += 1

                # Store node information
                for node_idx in range(n):
                    node_id = f"{graph_id}.{node_idx + 1}"
                    nodes.append({
                        'node_id': node_id,
                        'label': node_labels[node_idx]
                    })

                # Parse E (adjacency matrix)
                if i < len(lines) and lines[i] == 'E:':
                    i += 1
                    adj_matrix = []
                    for _ in range(n):
                        if i < len(lines):
                            row = list(map(int, lines[i].split()))
                            adj_matrix.append(row)
                            i += 1

                    # Extract edges (only upper triangle since undirected)
                    for src in range(n):
                        for dst in range(src + 1, n):
                            affinity = adj_matrix[src][dst]
                            if affinity > 0:  # Skip missing edges
                                is_laundering = max(node_labels[src], node_labels[dst])
                                src_id = f"{graph_id}.{src + 1}"
                                dst_id = f"{graph_id}.{dst + 1}"
                                edges.append({
                                    'src': src_id,
                                    'dst': dst_id,
                                    'is_laundering': is_laundering,
                                    'affinity': affinity
                                })
        else:
            i += 1

    return pd.DataFrame(edges), pd.DataFrame(nodes), graph_sizes
