import torch
import numpy as np
import pandas as pd
import os
import pathlib
import kagglehub
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import k_hop_subgraph, remove_self_loops, to_undirected
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from src.datasets.utils.financial_dataset_builder_utilities import DatasetBuilderUtilities
from src.datasets.utils.plots import plot_degree_distribution_comparison, plot_affinity_distribution, plot_degree_distribution_splits, plot_affinity_distribution_splits
from src.datasets.utils.sampling import sample_nodes_preserving_degree_distribution

RANDOM_STATE = 42
SAMPLING_FRACTION_NON_LAUNDERING_NODES = 0
SAMPLING_FRACTION_LAUNDERING_NODES = 0.5  # Fraction of laundering nodes to keep (0-1)
NUM_DEGREE_BINS = 100  # Number of bins for stratified degree sampling

class FinancialGraph(InMemoryDataset):

    def __init__(self, dataset_url, dataset_name, split, root, k_hop, transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        self.dataset_url = dataset_url
        self.dataset_name = dataset_name
        self.k_hop = k_hop

        # The maximum number of edges in a k-hop subgraph, used for edge_counts functions
        self.max_subgraph_edges = None
        if self.split == 'train':
            self.file_idx = 0
        elif self.split == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx], weights_only=False)

    @property
    def raw_file_names(self):
        return ['train.csv', 'val.csv', 'test.csv']

    @property
    def processed_file_names(self):
        return ['train_data.pt', 'val_data.pt', 'test_data.pt']

    def download(self):
        dataset_path = kagglehub.dataset_download(self.dataset_url, path=self.dataset_name)
        df = pd.read_csv(dataset_path)

        # Remove self-transactions and so the graph with only one node
        df = df[df['Account'] != df['Account.1']].reset_index(drop=True)

        # Keep only required columns
        df = df[['Account', 'Account.1', 'Is Laundering']].copy()

        # Group edges by (Source, Destination) and count occurrences
        df_grouped = df.groupby(['Account', 'Account.1'], as_index=False).agg({
            'Is Laundering': 'max'  # Keep laundering status (max ensures if any transaction was laundering, it's marked as such)
        })
        df_grouped['count'] = df.groupby(['Account', 'Account.1']).size().values

        # Categorize count into 5 quintile-based categories
        quintiles = df_grouped['count'].quantile([0.2, 0.4, 0.6, 0.8]).values

        def categorize_count(count_val):
            if count_val <= quintiles[0]:
                return 'very_low'
            elif count_val <= quintiles[1]:
                return 'low'
            elif count_val <= quintiles[2]:
                return 'medium'
            elif count_val <= quintiles[3]:
                return 'high'
            else:
                return 'very_high'

        df_grouped['affinity'] = df_grouped['count'].apply(categorize_count)
        df = df_grouped

        # Plot affinity distribution
        plot_path = os.path.join(self.raw_dir, 'affinity_distribution.png')
        plot_affinity_distribution(df['affinity'], plot_path)

        # Step 1: Get all unique accounts and their laundering status
        source_accounts = df[['Account', 'Is Laundering']]
        dest_accounts = df[['Account.1', 'Is Laundering']].rename(columns={'Account.1': 'Account'})
        all_accounts = pd.concat([source_accounts, dest_accounts])
        account_status = all_accounts.groupby('Account')['Is Laundering'].max().reset_index()

        stats_path = os.path.join(self.raw_dir, 'stats.txt')
        with open(stats_path, 'w') as f:
            f.write(f"Original dataset statistics:\n")
            f.write(f"  Total accounts: {len(account_status)}\n")
            f.write(f"  Laundering accounts: {account_status['Is Laundering'].sum()} ({account_status['Is Laundering'].sum()/len(account_status)*100:.2f}%)\n")
            f.write(f"  Total transactions: {len(df)}\n")
            f.write(f"  Laundering transactions: {len(df[df['Is Laundering'] == 1])} ({len(df[df['Is Laundering'] == 1])/len(df)*100:.2f}%)\n")

        laundering_accounts = account_status[account_status['Is Laundering'] == 1]

        # Sample laundering accounts while preserving degree distribution
        if SAMPLING_FRACTION_LAUNDERING_NODES < 1.0:
            original_laundering_count = len(laundering_accounts)

            # Count transactions per laundering account (as source or destination)
            laundering_degree_before = pd.concat([
                df[df['Account'].isin(laundering_accounts['Account'])]['Account'],
                df[df['Account.1'].isin(laundering_accounts['Account'])]['Account.1'].rename('Account')
            ]).value_counts()

            # Sample while preserving degree distribution
            laundering_accounts = sample_nodes_preserving_degree_distribution(
                laundering_accounts,
                laundering_degree_before,
                SAMPLING_FRACTION_LAUNDERING_NODES,
                NUM_DEGREE_BINS,
                RANDOM_STATE
            )

            # Get degrees after sampling
            laundering_degree_after = laundering_degree_before[laundering_degree_before.index.isin(laundering_accounts['Account'])]

            removed_count = original_laundering_count - len(laundering_accounts)
            removed_percentage = (removed_count / original_laundering_count * 100) if original_laundering_count > 0 else 0

            with open(stats_path, 'a') as f:
                f.write(f"\nAfter sampling laundering nodes (preserving degree distribution, {SAMPLING_FRACTION_LAUNDERING_NODES * 100:.1f}%):\n")
                f.write(f"  Remaining laundering accounts: {len(laundering_accounts)}\n")
                f.write(f"  Removed laundering accounts: {removed_count} ({removed_percentage:.2f}%)\n")

            # Plot degree distribution comparison
            plot_path = os.path.join(self.raw_dir, 'laundering_degree_distribution.png')
            plot_degree_distribution_comparison(
                laundering_degree_before,
                laundering_degree_after,
                'Laundering Nodes Degree Distribution Comparison',
                plot_path
            )

        # Step 2: Sample non-laundering accounts
        if SAMPLING_FRACTION_NON_LAUNDERING_NODES > 0 and SAMPLING_FRACTION_NON_LAUNDERING_NODES <= 1.0:
            non_laundering_accounts = account_status[account_status['Is Laundering'] == 0]

            # Sample only non-laundering accounts
            sampled_non_laundering, _ = train_test_split(
                non_laundering_accounts,
                train_size=SAMPLING_FRACTION_NON_LAUNDERING_NODES,
                random_state=RANDOM_STATE
            )

            # Combine all laundering + sampled non-laundering
            account_status = pd.concat([laundering_accounts, sampled_non_laundering]).reset_index(drop=True)
            sampled_ids = set(account_status['Account'])
            df = df[df['Account'].isin(sampled_ids) & df['Account.1'].isin(sampled_ids)].reset_index(drop=True)

            with open(stats_path, 'a') as f:
                f.write(f"\nAfter sampling ({SAMPLING_FRACTION_NON_LAUNDERING_NODES * 100:.1f}% of non-laundering accounts):\n")
                f.write(f"  Total accounts: {len(account_status)} (all {len(laundering_accounts)} laundering + {len(sampled_non_laundering)} non-laundering)\n")
                f.write(f"  Laundering accounts: {len(laundering_accounts)} ({len(laundering_accounts)/len(account_status)*100:.2f}%)\n")
                f.write(f"  Total transactions: {len(df)}\n")
                f.write(f"  Laundering transactions: {len(df[df['Is Laundering'] == 1])} ({len(df[df['Is Laundering'] == 1])/len(df)*100:.2f}%)\n")

        else:
            # Combine sampled laundering + sampled non-laundering and filter df by the sampled nodes
            account_status = laundering_accounts.reset_index(drop=True)
            sampled_ids = set(account_status['Account'])
            df = df[df['Account'].isin(sampled_ids) & df['Account.1'].isin(sampled_ids)].reset_index(drop=True)


        # Step 3: Update account_status to only include accounts that appear in df
        accounts_in_df = set(df['Account']).union(set(df['Account.1']))
        account_status = account_status[account_status['Account'].isin(accounts_in_df)].reset_index(drop=True)

        # Calculate degrees for all accounts (before split)
        account_degrees_before_split = pd.concat([
            df['Account'],
            df['Account.1'].rename('Account')
        ]).value_counts()

        # Add degree information to account_status
        account_status['degree'] = account_status['Account'].map(account_degrees_before_split)

        # Create degree bins using quantiles
        unique_degrees = account_status['degree'].nunique()
        effective_bins = min(NUM_DEGREE_BINS, unique_degrees)
        account_status['degree_bin'] = pd.qcut(
            account_status['degree'],
            q=effective_bins,
            labels=False,
            duplicates='drop'
        )

        # Step 4: Stratified split on accounts for train/val/test (preserving degree distribution)
        train_accounts, temp_accounts = train_test_split(
            account_status,
            test_size=0.4,
            stratify=account_status['degree_bin'],
            random_state=RANDOM_STATE
        )

        val_accounts, test_accounts = train_test_split(
            temp_accounts,
            test_size=0.5,
            stratify=temp_accounts['degree_bin'],
            random_state=RANDOM_STATE
        )

        # Step 5: Filter transactions where both accounts are in the split
        train_ids = set(train_accounts['Account'])
        val_ids = set(val_accounts['Account'])
        test_ids = set(test_accounts['Account'])

        train_df = df[df['Account'].isin(train_ids) & df['Account.1'].isin(train_ids)]
        val_df = df[df['Account'].isin(val_ids) & df['Account.1'].isin(val_ids)]
        test_df = df[df['Account'].isin(test_ids) & df['Account.1'].isin(test_ids)]

        # Calculate degrees for each split
        degrees_train = pd.concat([train_df['Account'], train_df['Account.1'].rename('Account')]).value_counts()
        degrees_val = pd.concat([val_df['Account'], val_df['Account.1'].rename('Account')]).value_counts()
        degrees_test = pd.concat([test_df['Account'], test_df['Account.1'].rename('Account')]).value_counts()

        # Plot degree distribution comparison across splits
        plot_path_degree = os.path.join(self.raw_dir, 'degree_distribution_splits.png')
        plot_degree_distribution_splits(
            account_degrees_before_split,
            degrees_train,
            degrees_val,
            degrees_test,
            plot_path_degree
        )

        # Plot affinity distribution comparison across splits
        plot_path_affinity = os.path.join(self.raw_dir, 'affinity_distribution_splits.png')
        plot_affinity_distribution_splits(
            df['affinity'],
            train_df['affinity'],
            val_df['affinity'],
            test_df['affinity'],
            plot_path_affinity
        )

        del temp_accounts, effective_bins, unique_degrees, account_degrees_before_split, account_status
        del degrees_train, degrees_val, degrees_test

        with open(stats_path, 'a') as f:
            f.write(f"\nTrain split:\n")
            f.write(f"  Accounts: {len(train_accounts)} ({train_accounts['Is Laundering'].sum()} laundering)\n")
            f.write(f"  Transactions: {len(train_df)} ({train_df['Is Laundering'].sum()} laundering)\n")

            f.write(f"\nVal split:\n")
            f.write(f"  Accounts: {len(val_accounts)} ({val_accounts['Is Laundering'].sum()} laundering)\n")
            f.write(f"  Transactions: {len(val_df)} ({val_df['Is Laundering'].sum()} laundering)\n")

            f.write(f"\nTest split:\n")
            f.write(f"  Accounts: {len(test_accounts)} ({test_accounts['Is Laundering'].sum()} laundering)\n")
            f.write(f"  Transactions: {len(test_df)} ({test_df['Is Laundering'].sum()} laundering)\n")

        # Save the split files
        train_df.to_csv(os.path.join(self.raw_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(self.raw_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(self.raw_dir, 'test.csv'), index=False)

    def process(self):
        affinity_categories = ['very_low', 'low', 'medium', 'high', 'very_high']
        file_path = os.path.join(self.raw_dir, f'{self.split}.csv')
        df = pd.read_csv(file_path)

        # Preprocess the data
        df.rename(columns={'Account': 'Source Account'}, inplace=True)
        df.rename(columns={'Account.1': 'Destination Account'}, inplace=True)

        edge_df, nodes_df = DatasetBuilderUtilities.preprocess(df, self.raw_paths[self.file_idx])

        # 1. Build a mapping from account IDs to node indices
        account_2idx = { acc: i for i, acc in enumerate(nodes_df['Account']) }

        # 2. Create edge_index tensor (2 x num_edges)
        src = edge_df['Source Account'].map(account_2idx).values
        dst = edge_df['Destination Account'].map(account_2idx).values
        edge_indexes = torch.tensor(np.array([src, dst]), dtype=torch.long)

        # 3. Prepare edge features (features for each transaction/edge)
        # Map affinity categories to indices
        affinity_to_idx = {cat: i+1 for i, cat in enumerate(affinity_categories)}  # +1 to reserve 0 for edge absence
        edge_features = edge_df['affinity'].map(affinity_to_idx)
        edge_features = torch.tensor(edge_features, dtype=torch.long)

        # 4. Make the graph undirected using PyTorch Geometric's to_undirected
        # Apply this BEFORE one-hot encoding so we can use reduce='max' on scalar values
        edge_indexes, edge_features = to_undirected(edge_indexes, edge_features, reduce='max')

        # 5. Now apply one-hot encoding
        edge_attr = F.one_hot(
            edge_features,
            num_classes=len(affinity_categories) + 1 # +1 because 0 label is reserved to indicate the edge absence
        ).to(torch.float)

        data_list: list[Data] = []
        avg_subgraph_size = 0

        # 5. Iterate over each account and build the k-hop subgraph
        for _, node_index in account_2idx.items():
            # 5.1 Build the k-hop subgraph around the account node
            subset_nodes, subgraph_edge_index, _, edge_mask = \
                k_hop_subgraph(node_index, self.k_hop, edge_indexes, relabel_nodes=True)
            N = len(subset_nodes)
            avg_subgraph_size += N

            assert N > 1
            assert (subgraph_edge_index[0] != subgraph_edge_index[1]).all(), "Self-loops detected in subgraph"

            # 5.2 Get the edge attributes for the subgraph
            subgraph_edge_attr = edge_attr[edge_mask]

            perm = (subgraph_edge_index[0] * N + subgraph_edge_index[1]).argsort()
            subgraph_edge_index = subgraph_edge_index[:, perm]
            subgraph_edge_attr = subgraph_edge_attr[perm]

            # 5.5 Create the true labels (Is Laundering) for the subgraph edges
            x = F.one_hot(
                torch.tensor(nodes_df.iloc[subset_nodes]['Is Laundering'].values, dtype=torch.long), 
                num_classes=2
            ).float()
            y = torch.zeros(size=(1, 0), dtype=torch.float)

            # 5.6 Create the Data object for the subgraph
            data = Data(
                x=x,
                edge_index=subgraph_edge_index,
                edge_attr=subgraph_edge_attr,
                y=y,
                num_nodes=N
            )
            data_list.append(data)

        avg_subgraph_size /= len(account_2idx)

        stats_path = os.path.join(self.raw_dir, 'stats.txt')
        with open(stats_path, 'a') as f:
            f.write(f"\n{self.split.capitalize()} k-hop subgraphs:\n")
            f.write(f"  Number of k-hop subgraphs: {len(data_list)}\n")
            f.write(f"  Average k-hop subgraph size: {avg_subgraph_size:.2f}\n")

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])


class FinancialGraphDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        datasets = {'train': FinancialGraph(dataset_url=self.cfg.dataset.datasource.url, 
                                                dataset_name=self.cfg.dataset.datasource.name,
                                                 split='train', root=root_path, k_hop=self.cfg.dataset.k_hop),
                    'val': FinancialGraph(dataset_url=self.cfg.dataset.datasource.url,
                                            dataset_name=self.cfg.dataset.datasource.name,
                                            split='val', root=root_path, k_hop=self.cfg.dataset.k_hop),
                    'test': FinancialGraph(dataset_url=self.cfg.dataset.datasource.url,
                                           dataset_name=self.cfg.dataset.datasource.name,
                                            split='test', root=root_path, k_hop=self.cfg.dataset.k_hop)}
        # print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]
    
class FinancialDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts(max_nodes_possible=706_000)
        self.node_types = torch.tensor([0.0, 1.0], dtype=torch.float)
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)