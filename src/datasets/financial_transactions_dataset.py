import torch
import numpy as np
import pandas as pd
import os
import pathlib
import kagglehub
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import k_hop_subgraph, remove_self_loops
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from src.datasets.utils.financial_dataset_builder_utilities import DatasetBuilderUtilities

RANDOM_STATE = 42
SAMPLING_FRACTION_NON_LAUNDERING_NODES = 0
MIN_DEGREE_LAUNDERING_NODES = 85  # Minimum number of transactions for laundering nodes

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

        # Step 1: Get all unique accounts and their laundering status
        source_accounts = df[['Account', 'Is Laundering']]
        dest_accounts = df[['Account.1', 'Is Laundering']].rename(columns={'Account.1': 'Account'})
        all_accounts = pd.concat([source_accounts, dest_accounts])
        account_status = all_accounts.groupby('Account')['Is Laundering'].max().reset_index()

        print(f"\nOriginal dataset statistics:")
        print(f"  Total accounts: {len(account_status)}")
        print(f"  Laundering accounts: {account_status['Is Laundering'].sum()} ({account_status['Is Laundering'].sum()/len(account_status)*100:.2f}%)")
        print(f"  Total transactions: {len(df)}")
        print(f"  Laundering transactions: {len(df[df['Is Laundering'] == 1])} ({len(df[df['Is Laundering'] == 1])/len(df)*100:.2f}%)")

        laundering_accounts = account_status[account_status['Is Laundering'] == 1]

        # Filter laundering accounts by degree (transaction count)
        if MIN_DEGREE_LAUNDERING_NODES > 0:
            original_laundering_count = len(laundering_accounts)

            # Count transactions per laundering account (as source or destination)
            laundering_degree = pd.concat([
                df[df['Account'].isin(laundering_accounts['Account'])]['Account'],
                df[df['Account.1'].isin(laundering_accounts['Account'])]['Account.1'].rename('Account')
            ]).value_counts()

            # Keep only laundering accounts with degree >= threshold
            high_degree_accounts = laundering_degree[laundering_degree >= MIN_DEGREE_LAUNDERING_NODES].index
            laundering_accounts = laundering_accounts[laundering_accounts['Account'].isin(high_degree_accounts)].reset_index(drop=True)

            removed_count = original_laundering_count - len(laundering_accounts)
            removed_percentage = (removed_count / original_laundering_count * 100) if original_laundering_count > 0 else 0

            print(f"\nAfter filtering laundering nodes by degree (>= {MIN_DEGREE_LAUNDERING_NODES} transactions):")
            print(f"  Remaining laundering accounts: {len(laundering_accounts)}")
            print(f"  Removed laundering accounts: {removed_count} ({removed_percentage:.2f}%)")

        # Step 2: Sample non-laundering accounts while keeping ALL laundering accounts
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

            print(f"\nAfter sampling ({SAMPLING_FRACTION_NON_LAUNDERING_NODES * 100:.1f}% of non-laundering accounts):")
            print(f"  Total accounts: {len(account_status)} (all {len(laundering_accounts)} laundering + {len(sampled_non_laundering)} non-laundering)")
            print(f"  Laundering accounts: {len(laundering_accounts)} ({len(laundering_accounts)/len(account_status)*100:.2f}%)")
            print(f"  Total transactions: {len(df)}")
            print(f"  Laundering transactions: {len(df[df['Is Laundering'] == 1])} ({len(df[df['Is Laundering'] == 1])/len(df)*100:.2f}%)")

        else:
            # Combine all laundering + sampled non-laundering
            account_status = laundering_accounts.reset_index(drop=True)
            sampled_ids = set(account_status['Account'])
            df = df[df['Account'].isin(sampled_ids) & df['Account.1'].isin(sampled_ids)].reset_index(drop=True)


        # Step 3: Stratified split on accounts for train/val/test
        train_accounts, temp_accounts = train_test_split(
            account_status,
            test_size=0.3,
            stratify=account_status['Is Laundering'],
            random_state=RANDOM_STATE
        )

        val_accounts, test_accounts = train_test_split(
            temp_accounts,
            test_size=0.5,
            stratify=temp_accounts['Is Laundering'],
            random_state=RANDOM_STATE
        )

        # Step 4: Filter transactions where both accounts are in the split
        train_ids = set(train_accounts['Account'])
        val_ids = set(val_accounts['Account'])
        test_ids = set(test_accounts['Account'])

        train_df = df[df['Account'].isin(train_ids) & df['Account.1'].isin(train_ids)]
        val_df = df[df['Account'].isin(val_ids) & df['Account.1'].isin(val_ids)]
        test_df = df[df['Account'].isin(test_ids) & df['Account.1'].isin(test_ids)]

        print(f"\nTrain split:")
        print(f"  Accounts: {len(train_accounts)} ({train_accounts['Is Laundering'].sum()} laundering)")
        print(f"  Transactions: {len(train_df)} ({train_df['Is Laundering'].sum()} laundering)")

        print(f"\nVal split:")
        print(f"  Accounts: {len(val_accounts)} ({val_accounts['Is Laundering'].sum()} laundering)")
        print(f"  Transactions: {len(val_df)} ({val_df['Is Laundering'].sum()} laundering)")

        print(f"\nTest split:")
        print(f"  Accounts: {len(test_accounts)} ({test_accounts['Is Laundering'].sum()} laundering)")
        print(f"  Transactions: {len(test_df)} ({test_df['Is Laundering'].sum()} laundering)")

        # Save the split files
        train_df.to_csv(os.path.join(self.raw_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(self.raw_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(self.raw_dir, 'test.csv'), index=False)

    def process(self):
        payment_formats = [ # this is for one hot encoding
            'US Dollar', 'Euro', 'Bitcoin', 'Yuan', 'Yen', 'UK Pound','Brazil Real', 'Australian Dollar', 
            'Rupee', 'Ruble', 'Canadian Dollar', 'Mexican Peso', 'Swiss Franc', 'Shekel', 'Saudi Riyal'
        ]
        file_path = os.path.join(self.raw_dir, f'{self.split}.csv')
        df = pd.read_csv(file_path)

        # Preprocess the data
        df.rename(columns={'Account': 'Source Account'}, inplace=True)
        df.rename(columns={'Account.1': 'Destination Account'}, inplace=True)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.drop_duplicates(inplace=True)

        edge_df, nodes_df = DatasetBuilderUtilities.preprocess(df, self.raw_paths[self.file_idx])

        # 1. The dataframe contains only the edges from one direction (src -> dst), 
        #      but torch_geometric requires both directions (src -> dst and dst -> src) to be present for undirected graphs.
        #    So we add to the dataframe the reverse edges.
        undirected_edges_df = DatasetBuilderUtilities.get_undirected_edges_df(edge_df)
        del edge_df

        # 2. Build a mapping from laundering account IDs to node indices
        account_2idx = { acc: i for i, acc in enumerate(nodes_df['Account']) }

        # 3. Create edge_index tensor (2 x num_edges)
        src = undirected_edges_df['Source Account'].map(account_2idx).values
        dst = undirected_edges_df['Destination Account'].map(account_2idx).values
        laundering_edge_indexes = torch.tensor(np.array([src, dst]), dtype=torch.long)

        # TODO: needs to be refactored to treat each feature as category (label encoded) then apply one_hot encoding
        # 4. Prepare edge features (features for each transaction/edge)
        # laundering_edge_features = undirected_laundering_edges_df[
        #     ['Amount Paid', 'Timestamp', 'hour', 'day of month', 'month', 'weekday',
        #     'Receiving Currency', 'Payment Currency', 'Payment Format']
        # ].values
        laundering_edge_features = undirected_edges_df['Payment Format'] + 1
        laundering_edge_features = torch.tensor(laundering_edge_features, dtype=torch.long)
        edge_attr = F.one_hot(
            laundering_edge_features,
            num_classes=len(payment_formats) + 1 # +1 because 0 label is reserved to indicate the edge absence
        ).to(torch.float)

        data_list: list[Data] = []
        avg_subgraph_size = 0

        # 5. Iterate over each laundering account and build the k-hop subgraph
        for _, node_index in account_2idx.items():
            # 5.1 Build the k-hop subgraph around the laundering account node
            subset_laundering_nodes, subgraph_edge_index, _, edge_mask = \
                k_hop_subgraph(node_index, self.k_hop, laundering_edge_indexes, relabel_nodes=True)
            N = len(subset_laundering_nodes)
            avg_subgraph_size += N

            assert N > 1

            # 5.2 Get the edge attributes for the subgraph
            subgraph_edge_attr_full = edge_attr[edge_mask]

            # 5.3 Remove self-loops from edge_index and edge_attr
            subgraph_edge_index, subgraph_edge_attr = remove_self_loops(
                subgraph_edge_index, subgraph_edge_attr_full
            )

            perm = (subgraph_edge_index[0] * N + subgraph_edge_index[1]).argsort()
            subgraph_edge_index = subgraph_edge_index[:, perm]
            subgraph_edge_attr = subgraph_edge_attr[perm]

            # 5.5 Create the true labels (Is Laundering) for the subgraph edges
            x = F.one_hot(
                torch.tensor(nodes_df.iloc[subset_laundering_nodes]['Is Laundering'].values, dtype=torch.long), 
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
        print(f"Number of k-hop subgraphs: [{len(data_list)}]")
        print(f'Average k-hop subgraph size: [{avg_subgraph_size:.2f}]')

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