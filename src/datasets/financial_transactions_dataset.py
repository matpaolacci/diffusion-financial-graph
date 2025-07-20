import torch
import numpy as np
import pandas as pd
import os
import pathlib

from torch_geometric.data import Data, InMemoryDataset
from sklearn.model_selection import train_test_split
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from graph_tool.all import label_components, Graph, GraphView

RANDOM_STATE = 42

class DatasetBuilderUtilities:
    @staticmethod
    def _get_largest_component_df(edges_df: pd.DataFrame, csv_path: str) -> pd.DataFrame:
        """
        Extract the largest component of the graph.
        :param edges_df: The dataframe with the edges of the graph.
        :return: A DataFrame containing only the edges from the largest component.
        """
        print(f'Processing df from [{csv_path}]')
        
        trans_graph = Graph(
            list(edges_df[['Source Account', 'Destination Account']].itertuples(index=False, name=None)),
            hashed=True,
            directed=True
        )

        comp_label, hist = label_components(trans_graph, directed=False)
        largest_comp_label = hist.argmax()

        # Create a filter to mask trans_graph and obtain the largest component's graph
        vfilt = trans_graph.new_vertex_property("bool")
        vfilt.a = (comp_label.a == largest_comp_label)

        largest_comp_view = GraphView(trans_graph, vfilt=vfilt)
        print(f'Original graph has [{trans_graph.num_vertices()}] vertices')
        print(f'Largest component has [{largest_comp_view.num_vertices()}] vertices')
        print(f'Original graph has [{trans_graph.num_edges()}] edges')
        print(f'Largest component has [{largest_comp_view.num_edges()}] edges')

        # Get the string IDs of the vertices from the internal property map
        vertex_ids = trans_graph.vp.ids

        # Extract edges from the largest component view
        largest_comp_edges = set()
        for edge in largest_comp_view.edges():
            source_id = vertex_ids[edge.source()]
            dest_id = vertex_ids[edge.target()]
            largest_comp_edges.add((source_id, dest_id))

        # Create a DataFrame with the edges from the largest component
        edges_largest_comp_df = pd.DataFrame(list(largest_comp_edges), columns=['Source Account', 'Destination Account'])

        # Preserve order of edges_df by using its index
        edges_df_with_index = edges_df.reset_index()

        # Filter the original edges_df to keep only the edges present in the largest component
        filtered_edges_df = pd.merge(edges_df_with_index, edges_largest_comp_df, on=['Source Account', 'Destination Account'], how='inner')

        # Sort by the original index to restore order, then drop the index column
        filtered_edges_df_sorted = filtered_edges_df.sort_values('index').drop(columns=['index'])
        print(f'Original dataframe has [{edges_df.shape[0]}] edges')
        print(f'Filtered dataframe has [{filtered_edges_df_sorted.shape[0]}] edges')

        return filtered_edges_df_sorted

    @staticmethod
    def _get_nodes(df: pd.DataFrame) -> pd.DataFrame:
        ldf = df[['Source Account', 'From Bank']]
        rdf = df[['Destination Account', 'To Bank']] 

        # Get all illicit transactions
        suspicious = df[df['Is Laundering']==1]

        # Separate source and destination accounts involved in illicit transactions.
        source_df = suspicious[['Source Account', 'Is Laundering']].rename({'Source Account': 'Account'}, axis=1)
        destination_df = suspicious[['Destination Account', 'Is Laundering']].rename({'Destination Account': 'Account'}, axis=1)

        # Joint into a single DataFrame
        suspicious = pd.concat([source_df, destination_df], join='outer')

        # An account could be involved in several transactions, so we drop duplicates
        suspicious = suspicious.drop_duplicates()

        # Merge the source and destination accounts with their respective banks
        ldf = ldf.rename({'Source Account': 'Account', 'From Bank': 'Bank'}, axis=1)
        rdf = rdf.rename({'Destination Account': 'Account', 'To Bank': 'Bank'}, axis=1)
        df = pd.concat([ldf, rdf], join='outer')
        df = df.drop_duplicates()

        df['Is Laundering'] = 0

        # Mark all the transactions of the accounts involved in illicit transactions as illicit
        df.set_index('Account', inplace=True)
        df.update(suspicious.set_index('Account'))
        return df.reset_index()

    @staticmethod
    def preprocess(df: pd.DataFrame, csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
        def df_label_encoder(df, columns):
            from sklearn import preprocessing
            le = preprocessing.LabelEncoder()
            for i in columns:
                df[i] = le.fit_transform(df[i].astype(str))
            return df

        # Extract some features from the 'Timestamp'
        df['hour'] = df['Timestamp'].dt.hour
        df['day of month'] = df['Timestamp'].dt.day
        df['month'] = df['Timestamp'].dt.month
        df['weekday'] = df['Timestamp'].dt.weekday
        
        # Put the 'Is Laundering' as last column
        cols = df.columns.tolist()
        cols.remove('Is Laundering')
        idx = cols.index('weekday') + 1
        cols.insert(idx, 'Is Laundering')
        df = df[cols]
        
        df = df_label_encoder(df,['Payment Format', 'Payment Currency', 'Receiving Currency'])
        
        # Scale the Timestamp feature to a real-valued range between 0 and 1 using min-max normalization
        df['Timestamp'] = df['Timestamp'].apply(lambda x: x.value)
        df['Timestamp'] = (df['Timestamp']-df['Timestamp'].min())/(df['Timestamp'].max()-df['Timestamp'].min())

        df['Source Account'] = df['From Bank'].astype(str) + '_' + df['Source Account']
        df['Destination Account'] = df['To Bank'].astype(str) + '_' + df['Destination Account']
        edges_df = df.sort_values(by=['Source Account'])
        
        largest_comp_edges_df = DatasetBuilderUtilities._get_largest_component_df(edges_df, csv_path)

        return largest_comp_edges_df, DatasetBuilderUtilities._get_nodes(largest_comp_edges_df)

class FinancialGraph(InMemoryDataset):

    def __init__(self, dataset_url, dataset_name, split, root, transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        self.dataset_url = dataset_url
        self.dataset_name = dataset_name
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
        df = pd.read_csv(dataset_path)

        # Stratified split based on the target column
        train_df, temp_df = train_test_split(
            df, test_size=0.7, stratify=df['Is Laundering'], random_state=RANDOM_STATE)

        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df['Is Laundering'], random_state=RANDOM_STATE)

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

        edges_df, all_accounts = DatasetBuilderUtilities.preprocess(df, self.raw_paths[self.file_idx])

        # 1. Build a mapping from account IDs to node indices
        account2idx = {acc: i for i, acc in enumerate(all_accounts['Account'])}

        # 2. Create edge_index tensor (2 x num_edges)
        src = edges_df['Source Account'].map(account2idx).values
        dst = edges_df['Destination Account'].map(account2idx).values
        edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

        # 3. Prepare edge features (features for each transaction/edge)
        edge_features = edges_df[
            ['Amount Paid', 'Timestamp', 'hour', 'day of month', 'month', 'weekday',
            'Receiving Currency', 'Payment Currency', 'Payment Format']
        ].values
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

        # 4. Labels (target variable for each edge)
        y = torch.tensor(edges_df['Is Laundering'].values, dtype=torch.long)

        # need to add node features
        x = torch.zeros((len(all_accounts), 1)) 

        # 5. Build the PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=len(all_accounts)
        )

        torch.save(self.collate([data]), self.processed_paths[self.file_idx])


class FinancialGraphDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        datasets = {'train': FinancialGraph(dataset_url=self.cfg.dataset.datasource.url, 
                                                dataset_name=self.cfg.dataset.datasource.name,
                                                 split='train', root=root_path),
                    'val': FinancialGraph(dataset_url=self.cfg.dataset.datasource.url,
                                            dataset_name=self.cfg.dataset.datasource.name,
                                            split='val', root=root_path),
                    'test': FinancialGraph(dataset_url=self.cfg.dataset.datasource.url,
                                           dataset_name=self.cfg.dataset.datasource.name,
                                            split='test', root=root_path)}
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
        self.node_types = torch.tensor([1])               # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)