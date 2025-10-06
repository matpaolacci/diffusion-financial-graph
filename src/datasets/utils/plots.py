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
    axes[0].hist(log_degrees_before, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('log2(Degree)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Before Sampling (n={len(degrees_before)})')
    axes[0].grid(True, alpha=0.3)

    # Histogram after sampling
    axes[1].hist(log_degrees_after, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_xlabel('log2(Degree)')
    axes[1].set_ylabel('Frequency')
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

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(categories, counts, color=['#d62728', '#ff7f0e', '#ffdd57', '#2ca02c', '#1f77b4'],
                   edgecolor='black', alpha=0.8)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel('Affinity Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Edges', fontsize=12, fontweight='bold')
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
    axes[0, 0].hist(log_degrees_before, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[0, 0].set_xlabel('log2(Degree)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Before Split (n={len(degrees_before)})')
    axes[0, 0].grid(True, alpha=0.3)

    # Histogram train split
    axes[0, 1].hist(log_degrees_train, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].set_xlabel('log2(Degree)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Train Split (n={len(degrees_train)})')
    axes[0, 1].grid(True, alpha=0.3)

    # Histogram val split
    axes[1, 0].hist(log_degrees_val, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].set_xlabel('log2(Degree)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Val Split (n={len(degrees_val)})')
    axes[1, 0].grid(True, alpha=0.3)

    # Histogram test split
    axes[1, 1].hist(log_degrees_test, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].set_xlabel('log2(Degree)')
    axes[1, 1].set_ylabel('Frequency')
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
        bars = ax.bar(categories, counts, color=colors, edgecolor='black', alpha=0.8)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Affinity Category')
        ax.set_ylabel('Number of Edges')
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
