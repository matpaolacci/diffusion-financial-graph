import pandas as pd

def sample_nodes_preserving_degree_distribution(accounts_df, degrees, sampling_fraction, num_bins, random_state):
    """
    Sample accounts while preserving the degree distribution using stratified sampling.

    Args:
        accounts_df: DataFrame with account information
        degrees: Series with account degrees (index=account, value=degree)
        sampling_fraction: Fraction of accounts to sample (0-1)
        num_bins: Number of degree bins for stratification
        random_state: Random state for reproducibility

    Returns:
        Sampled accounts DataFrame
    """
    if sampling_fraction >= 1.0:
        return accounts_df

    if sampling_fraction <= 0:
        return pd.DataFrame(columns=accounts_df.columns)

    # Add degree information to accounts
    accounts_with_degree = accounts_df.copy()
    accounts_with_degree['degree'] = accounts_with_degree['Account'].map(degrees)

    # Print unique degree values
    unique_degrees = accounts_with_degree['degree'].nunique()
    print(f"  Number of unique degree values: {unique_degrees}")
    print(f"  Requested bins: {num_bins}, effective bins: {min(num_bins, unique_degrees)}")

    # Create degree bins using quantiles for balanced stratification
    accounts_with_degree['degree_bin'] = pd.qcut(
        accounts_with_degree['degree'],
        q=num_bins,
        labels=False,
        duplicates='drop'
    )

    # Stratified sampling by degree bin
    sampled_accounts = []
    for bin_id in accounts_with_degree['degree_bin'].unique():
        bin_accounts = accounts_with_degree[accounts_with_degree['degree_bin'] == bin_id]
        sample_size = max(1, int(len(bin_accounts) * sampling_fraction))

        sampled_bin = bin_accounts.sample(n=sample_size, random_state=random_state)
        sampled_accounts.append(sampled_bin)

    result = pd.concat(sampled_accounts).drop(columns=['degree', 'degree_bin']).reset_index(drop=True)
    return result