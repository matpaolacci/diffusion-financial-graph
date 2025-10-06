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
