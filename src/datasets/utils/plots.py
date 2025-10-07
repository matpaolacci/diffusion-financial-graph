import matplotlib.pyplot as plt
import numpy as np

def plot_degree_distribution_comparison(degrees_before, degrees_after, title, save_path):
    """
    Plot histograms comparing degree distributions before and after sampling.

    Args:
        degrees_before: Series with degrees before sampling
        degrees_after: Series with degrees after sampling
        title: Title for the plot
        save_path: Path to save the figure
    """
    # Convert degrees to log2 scale
    log_degrees_before = np.log2(degrees_before)
    log_degrees_after = np.log2(degrees_after)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram before sampling
    weights_before = np.ones_like(log_degrees_before) / len(log_degrees_before) * 100
    axes[0].hist(log_degrees_before, bins=50, weights=weights_before, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('log2(Degree)')
    axes[0].set_ylabel('Percentage (%)')
    axes[0].set_title(f'Before Sampling (n={len(degrees_before)})')
    axes[0].grid(True, alpha=0.3)

    # Histogram after sampling
    weights_after = np.ones_like(log_degrees_after) / len(log_degrees_after) * 100
    axes[1].hist(log_degrees_after, bins=50, weights=weights_after, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_xlabel('log2(Degree)')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_title(f'After Sampling (n={len(degrees_after)})')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Degree distribution plot saved to: {save_path}")


def plot_affinity_distribution(affinity_series, save_path):
    """
    Plot bar chart of affinity distribution.

    Args:
        affinity_series: Pandas Series with affinity categories
        save_path: Path to save the figure
    """
    # Count occurrences of each affinity category
    affinity_counts = affinity_series.value_counts()

    # Define the order of categories
    categories = ['very_low', 'low', 'medium', 'high', 'very_high']
    counts = [affinity_counts.get(cat, 0) for cat in categories]

    # Convert to percentages
    total = sum(counts)
    percentages = [(count / total * 100) if total > 0 else 0 for count in counts]

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(categories, percentages, color=['#d62728', '#ff7f0e', '#ffdd57', '#2ca02c', '#1f77b4'],
                   edgecolor='black', alpha=0.8)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel('Affinity Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Edge Affinity Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Affinity distribution plot saved to: {save_path}")


def plot_degree_distribution_splits(degrees_before, degrees_train, degrees_val, degrees_test, save_path):
    """
    Plot degree distributions comparing original dataset and train/val/test splits.

    Args:
        degrees_before: Series with degrees before split
        degrees_train: Series with degrees in train split
        degrees_val: Series with degrees in val split
        degrees_test: Series with degrees in test split
        save_path: Path to save the figure
    """
    # Convert degrees to log2 scale
    log_degrees_before = np.log2(degrees_before)
    log_degrees_train = np.log2(degrees_train)
    log_degrees_val = np.log2(degrees_val)
    log_degrees_test = np.log2(degrees_test)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogram before split
    weights_before = np.ones_like(log_degrees_before) / len(log_degrees_before) * 100
    axes[0, 0].hist(log_degrees_before, bins=50, weights=weights_before, alpha=0.7, color='purple', edgecolor='black')
    axes[0, 0].set_xlabel('log2(Degree)')
    axes[0, 0].set_ylabel('Percentage (%)')
    axes[0, 0].set_title(f'Before Split (n={len(degrees_before)})')
    axes[0, 0].grid(True, alpha=0.3)

    # Histogram train split
    weights_train = np.ones_like(log_degrees_train) / len(log_degrees_train) * 100
    axes[0, 1].hist(log_degrees_train, bins=50, weights=weights_train, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].set_xlabel('log2(Degree)')
    axes[0, 1].set_ylabel('Percentage (%)')
    axes[0, 1].set_title(f'Train Split (n={len(degrees_train)})')
    axes[0, 1].grid(True, alpha=0.3)

    # Histogram val split
    weights_val = np.ones_like(log_degrees_val) / len(log_degrees_val) * 100
    axes[1, 0].hist(log_degrees_val, bins=50, weights=weights_val, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].set_xlabel('log2(Degree)')
    axes[1, 0].set_ylabel('Percentage (%)')
    axes[1, 0].set_title(f'Val Split (n={len(degrees_val)})')
    axes[1, 0].grid(True, alpha=0.3)

    # Histogram test split
    weights_test = np.ones_like(log_degrees_test) / len(log_degrees_test) * 100
    axes[1, 1].hist(log_degrees_test, bins=50, weights=weights_test, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].set_xlabel('log2(Degree)')
    axes[1, 1].set_ylabel('Percentage (%)')
    axes[1, 1].set_title(f'Test Split (n={len(degrees_test)})')
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle('Degree Distribution Comparison Across Splits', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Degree distribution splits plot saved to: {save_path}")


def plot_affinity_distribution_splits(affinity_before, affinity_train, affinity_val, affinity_test, save_path):
    """
    Plot affinity distributions comparing original dataset and train/val/test splits.

    Args:
        affinity_before: Series with affinity before split
        affinity_train: Series with affinity in train split
        affinity_val: Series with affinity in val split
        affinity_test: Series with affinity in test split
        save_path: Path to save the figure
    """
    categories = ['very_low', 'low', 'medium', 'high', 'very_high']
    colors = ['#d62728', '#ff7f0e', '#ffdd57', '#2ca02c', '#1f77b4']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Helper function to plot bar chart
    def plot_affinity(ax, affinity_series, title, color_set):
        affinity_counts = affinity_series.value_counts()
        counts = [affinity_counts.get(cat, 0) for cat in categories]

        # Convert to percentages
        total = sum(counts)
        percentages = [(count / total * 100) if total > 0 else 0 for count in counts]

        bars = ax.bar(categories, percentages, color=colors, edgecolor='black', alpha=0.8)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Affinity Category')
        ax.set_ylabel('Percentage (%)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

    # Plot all four subplots
    plot_affinity(axes[0, 0], affinity_before, f'Before Split (n={len(affinity_before)})', colors)
    plot_affinity(axes[0, 1], affinity_train, f'Train Split (n={len(affinity_train)})', colors)
    plot_affinity(axes[1, 0], affinity_val, f'Val Split (n={len(affinity_val)})', colors)
    plot_affinity(axes[1, 1], affinity_test, f'Test Split (n={len(affinity_test)})', colors)

    fig.suptitle('Affinity Distribution Comparison Across Splits', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Affinity distribution splits plot saved to: {save_path}")


def plot_khop_subgraph_distributions(subgraph_node_sizes, subgraph_edge_counts, split_name, save_path):
    """
    Plot node and edge distributions for k-hop subgraphs.

    Args:
        subgraph_node_sizes: List of node counts for each subgraph
        subgraph_edge_counts: List of edge counts for each subgraph
        split_name: Name of the split (e.g., 'train', 'val', 'test')
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram for node distribution
    weights_nodes = np.ones(len(subgraph_node_sizes)) / len(subgraph_node_sizes) * 100
    axes[0].hist(subgraph_node_sizes, bins=30, weights=weights_nodes, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Node Distribution (n={len(subgraph_node_sizes)})', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Histogram for edge distribution
    weights_edges = np.ones(len(subgraph_edge_counts)) / len(subgraph_edge_counts) * 100
    axes[1].hist(subgraph_edge_counts, bins=30, weights=weights_edges, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_xlabel('Number of Edges', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Edge Distribution (n={len(subgraph_edge_counts)})', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f'{split_name.capitalize()} K-hop Subgraph Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  K-hop subgraph distributions plot saved to: {save_path}")


def plot_density_distribution(densities, split_name, save_path):
    """
    Plot graph density distribution for a single split, divided into 10 quantiles (deciles).

    Args:
        densities: List or array of density values
        split_name: Name of the split (e.g., 'train', 'val', 'test')
        save_path: Path to save the figure
    """
    densities_arr = np.array(densities)

    # Calculate 10 quantiles (deciles)
    quantiles = np.percentile(densities_arr, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    # Assign each density to a quantile bin
    quantile_labels = ['Q1\n(0-10%)', 'Q2\n(10-20%)', 'Q3\n(20-30%)', 'Q4\n(30-40%)', 'Q5\n(40-50%)',
                       'Q6\n(50-60%)', 'Q7\n(60-70%)', 'Q8\n(70-80%)', 'Q9\n(80-90%)', 'Q10\n(90-100%)']
    quantile_counts = [0] * 10

    for density in densities_arr:
        for i in range(10):
            if quantiles[i] <= density <= quantiles[i + 1]:
                quantile_counts[i] += 1
                break

    # Convert to percentages
    total = sum(quantile_counts)
    percentages = [(count / total * 100) if total > 0 else 0 for count in quantile_counts]

    # Define color based on split name
    color_map = {'train': 'blue', 'val': 'green', 'test': 'orange', 'dataset': 'purple'}
    color = color_map.get(split_name.lower(), 'gray')

    # Create bar plot
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(quantile_labels, percentages, color=color, edgecolor='black', alpha=0.7)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Density Quantile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{split_name.capitalize()} - Density Distribution (n={len(densities)})',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Density distribution plot saved to: {save_path}")
