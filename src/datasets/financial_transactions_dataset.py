import torch
import numpy as np
import pandas as pd
import os
import pathlib
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import k_hop_subgraph, remove_self_loops
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from src.datasets.utils.financial_dataset_builder_utilities import DatasetBuilderUtilities

RANDOM_STATE = 42

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
        import kagglehub
        from sklearn.model_selection import train_test_split

        dataset_path = kagglehub.dataset_download(self.dataset_url, path=self.dataset_name)
        df = pd.read_csv(dataset_path).head(2_000_000)  # TODO: For debugging purposes, we use only 10k rows

        # A stratified split is used to ensure that the proportion of values in the sample
        #   produced is the same as the proportion of values produced in the whole dataset.
        train_df, temp_df = train_test_split(
            df, 
            test_size=0.3, 
            stratify=df['Is Laundering'], 
            random_state=RANDOM_STATE
        )

        val_df, test_df = train_test_split(
            temp_df, 
            test_size=0.5, 
            stratify=temp_df['Is Laundering'], 
            random_state=RANDOM_STATE
        )

        # Save the split files in the raw directory
        train_df.to_csv(os.path.join(self.raw_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(self.raw_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(self.raw_dir, 'test.csv'), index=False)

    def process(self):
        file_path = os.path.join(self.raw_dir, f'{self.split}.csv')
        df = pd.read_csv(file_path)

        # Preprocess the data
        df.rename(columns={'Account': 'Source Account'}, inplace=True)
        df.rename(columns={'Account.1': 'Destination Account'}, inplace=True)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.drop_duplicates(inplace=True)

        # 	TODO: It needs to add some random non laundering accounts to the dataset, 
        #       for now, we only have subgraphs around laundering accounts.
        laundering_edge_df, laundering_accounts = DatasetBuilderUtilities.preprocess(df, self.raw_paths[self.file_idx])
        
        # 1. The dataframe contains only the edges from one direction (src -> dst), 
        #      but torch_geometric requires both directions (src -> dst and dst -> src) to be present for undirected graphs.
        #    So we add to the dataframe the reverse edges.
        undirected_laundering_edges_df = DatasetBuilderUtilities.get_undirected_edges_df(laundering_edge_df)
        del laundering_edge_df

        # 2. Build a mapping from laundering account IDs to node indices
        laundering_account_2idx = { acc: i for i, acc in enumerate(laundering_accounts['Account']) }

        # 3. Create edge_index tensor (2 x num_edges)
        src = undirected_laundering_edges_df['Source Account'].map(laundering_account_2idx).values
        dst = undirected_laundering_edges_df['Destination Account'].map(laundering_account_2idx).values
        laundering_edge_indexes = torch.tensor(np.array([src, dst]), dtype=torch.long)

        # 4. Prepare edge features (features for each transaction/edge)
        laundering_edge_features = undirected_laundering_edges_df[
            ['Amount Paid', 'Timestamp', 'hour', 'day of month', 'month', 'weekday',
            'Receiving Currency', 'Payment Currency', 'Payment Format'] # TODO: Maybe it could be useful to add the Is Laundering property also here
        ].values
        laundering_edge_attr = torch.tensor(laundering_edge_features, dtype=torch.float)

        data_list: list[Data] = []
        avg_subgraph_size = 0
        max_subgraph_edges = 0

        # 5. Iterate over each laundering account and build the k-hop subgraph
        for _, node_index in laundering_account_2idx.items():
            # 5.1 Build the k-hop subgraph around the laundering account node
            subset_laundering_nodes, subgraph_edge_index, _, edge_mask = \
                k_hop_subgraph(node_index, self.k_hop, laundering_edge_indexes, relabel_nodes=True)
            avg_subgraph_size += len(subset_laundering_nodes)

            # 5.2 Get the edge attributes for the subgraph
            subgraph_edge_attr_full = laundering_edge_attr[edge_mask]

            # 5.3 Remove self-loops from edge_index and edge_attr
            subgraph_edge_index, subgraph_edge_attr_full = remove_self_loops(
                subgraph_edge_index, subgraph_edge_attr_full
            )

            max_subgraph_edges = max(max_subgraph_edges, subgraph_edge_index.shape[1])
            
            # 5.4 Get the subset of edge attributes corresponding to the subgraph edges
            subgraph_edge_attr = subgraph_edge_attr_full[:, :]

            # 5.5 Create the true labels (Is Laundering) for the subgraph edges
            x = F.one_hot(
                torch.tensor(laundering_accounts.iloc[subset_laundering_nodes]['Is Laundering'].values), 
                num_classes=2
            ).float()
            y = torch.zeros(size=(1, 0), dtype=torch.float)

            # 5.6 Create the Data object for the subgraph
            data = Data(
                x=x,
                edge_index=subgraph_edge_index,
                edge_attr=subgraph_edge_attr,
                y=y,
                num_nodes=subset_laundering_nodes.shape[0]
            )
            data_list.append(data)

        avg_subgraph_size /= len(laundering_account_2idx)
        print(f"Number of k-hop subgraphs: [{len(data_list)}]")
        print(f'Average k-hop subgraph size: [{avg_subgraph_size:.2f}]')

        self.max_subgraph_edges = max_subgraph_edges
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
    
    def edge_counts(self):
        num_classes = None
        for data in self.train_dataloader():
            num_classes = data.edge_attr.shape[1]
            break

        d = torch.zeros(num_classes, dtype=torch.float)

        # Iterates over each batch in the training dataloader
        for i, data in enumerate(self.train_dataloader()):
            num_edges = data.edge_index.shape[1]
            num_non_edges = (self.train_dataset.max_subgraph_edges * data.num_graphs) - num_edges

            # Adds up the values of each edge feature across all edges in the current batch
            edge_types = data.edge_attr.sum(dim=0)
            assert num_non_edges >= 0

            # It adds num_non_edges to d[0] to treat the absence of an edge as a specific feature that the model needs to learn.
            d[0] += num_non_edges
            
            #stores the sum of the values for each edge feature column
            d[1:] += edge_types[1:] 

        d = d / d.sum() # Convert to distribution
        return d
    
class FinancialDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts(max_nodes_possible=706_000)
        self.node_types = torch.tensor([0.0, 1.0], dtype=torch.float)
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)