import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from parse.final_sampling import parse_graph_file

def plot_degree_distribution(df, save_path='degree_distribution.png'):
    """
    Plot degree distribution of the parsed graph dataset.

    Args:
        df: DataFrame with columns src, dst, is_laundering, affinity
        save_path: Path to save the figure
    """
    # Calculate degree for each node
    src_counts = df['src'].value_counts()
    dst_counts = df['dst'].value_counts()

    # Combine source and destination counts
    all_nodes = set(df['src']).union(set(df['dst']))
    degrees = {}
    for node in all_nodes:
        degrees[node] = src_counts.get(node, 0) + dst_counts.get(node, 0)

    # Convert to log2 scale
    degree_values = list(degrees.values())
    log_degrees = np.log2(degree_values)

    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(log_degrees, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('log2(Degree)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'Degree Distribution (n={len(degree_values)})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Degree distribution plot saved to: {save_path}")


def plot_affinity_distribution(df, save_path='affinity_distribution.png'):
    """
    Plot affinity distribution of the parsed graph dataset.

    Args:
        df: DataFrame with columns src, dst, is_laundering, affinity
        save_path: Path to save the figure
    """
    # Count occurrences of each affinity value
    affinity_counts = df['affinity'].value_counts().sort_index()

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#d62728', '#ff7f0e', '#ffdd57', '#2ca02c', '#1f77b4']
    bars = ax.bar(affinity_counts.index.astype(str), affinity_counts.values,
                   color=[colors[i-1] if i <= 5 else '#888888' for i in affinity_counts.index],
                   edgecolor='black', alpha=0.8)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel('Affinity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Edges', fontsize=12, fontweight='bold')
    ax.set_title('Edge Affinity Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Affinity distribution plot saved to: {save_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analysis.py <graph_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    edges_df, nodes_df, graph_sizes = parse_graph_file(file_path)

    print(edges_df)
    print(f"\nTotal edges: {len(edges_df)}")
    print(f"Total nodes: {len(nodes_df)}")

    # Calculate statistics
    laundering_nodes = nodes_df[nodes_df['label'] == 1]
    num_laundering = len(laundering_nodes)
    num_total_nodes = len(nodes_df)
    laundering_percentage = (num_laundering / num_total_nodes * 100) if num_total_nodes > 0 else 0

    avg_graph_size = np.mean(graph_sizes) if graph_sizes else 0

    # Create output directory if it doesn't exist
    output_dir = "result_financial_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Write statistics to file
    stats_path = os.path.join(output_dir, "stats.txt")
    with open(stats_path, 'w') as f:
        f.write("Dataset Statistics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total graphs: {len(graph_sizes)}\n")
        f.write(f"Total nodes: {num_total_nodes}\n")
        f.write(f"Total edges: {len(edges_df)}\n\n")
        f.write(f"Laundering accounts: {num_laundering} ({laundering_percentage:.2f}%)\n")
        f.write(f"Non-laundering accounts: {num_total_nodes - num_laundering} ({100 - laundering_percentage:.2f}%)\n\n")
        f.write(f"Average graph size: {avg_graph_size:.2f} nodes\n")
        f.write(f"Min graph size: {min(graph_sizes) if graph_sizes else 0} nodes\n")
        f.write(f"Max graph size: {max(graph_sizes) if graph_sizes else 0} nodes\n")

    print(f"\nStatistics saved to: {stats_path}")

    # Generate plots
    plot_degree_distribution(edges_df, os.path.join(output_dir, "degree_distribution.png"))
    plot_affinity_distribution(edges_df, os.path.join(output_dir, "affinity_distribution.png"))
