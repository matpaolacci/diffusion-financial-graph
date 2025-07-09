import torch
from torch_geometric.data import Data, InMemoryDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
import pathlib

RANDOM_STATE = 42

class DatasetBuilderUtilities:
    @staticmethod
    def get_nodes(df: pd.DataFrame) -> pd.DataFrame:
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
    def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
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
        receiving_df = edges_df[['Destination Account', 'Amount Received', 'Receiving Currency']]
        paying_df = edges_df[['Source Account', 'Amount Paid', 'Payment Currency']]
        currency_ls = sorted(edges_df['Receiving Currency'].unique())

        return edges_df, receiving_df, paying_df, currency_ls

class FinancialGraph(InMemoryDataset):

    initialized = False

    def __init__(self, dataset_url, dataset_name, split, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.split = split
        self.dataset_url = dataset_url
        self.dataset_name = dataset_name
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.csv', 'val.csv', 'test.csv']

    @property
    def processed_file_names(self):
        return [f'{self.split}_data.pt']

    def download(self):
        if self.initialized:
            return

        import kagglehub
        from sklearn.model_selection import train_test_split

        dataset_path = kagglehub.dataset_download(self.dataset_url + "/" + self.dataset_name)
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

        self.initialized = True

    def process(self):
        file_path = os.path.join(self.raw_dir, f'{self.split}.csv')
        df = pd.read_csv(file_path)

        # Preprocess the data
        df.rename(columns={'Account': 'Source Account'}, inplace=True)
        df.rename(columns={'Account.1': 'Destination Account'}, inplace=True)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.drop_duplicates(inplace=True)

        edges_df, _, _, _ = DatasetBuilderUtilities.preprocess(df)
        all_accounts = DatasetBuilderUtilities.get_nodes(edges_df)

        # 1. Build a mapping from account IDs to node indices
        account2idx = {acc: i for i, acc in enumerate(all_accounts['Account'])}

        # 2. Create edge_index tensor (2 x num_edges)
        src = edges_df['Source Account'].map(account2idx).values
        dst = edges_df['Destination Account'].map(account2idx).values
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        # 3. Prepare edge features (features for each transaction/edge)
        edge_features = edges_df[
            ['Amount Paid', 'Timestamp', 'hour', 'day of month', 'month', 'weekday',
            'Receiving Currency', 'Payment Currency', 'Payment Format']
        ].values
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

        # 4. Labels (target variable for each edge)
        y = torch.tensor(edges_df['Is Laundering'].values, dtype=torch.long)

        # 5. Build the PyG Data object
        data = Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=len(all_accounts)
        )

        torch.save(self.collate(data), self.processed_paths[0])


class FinancialGraphDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        datasets = {'train': FinancialGraph(dataset_url=self.cfg.dataset.url, 
                                                dataset_name=self.cfg.dataset.name,
                                                 split='train', root=root_path),
                    'val': FinancialGraph(dataset_url=self.cfg.dataset.url,
                                            dataset_name=self.cfg.dataset.name,
                                            split='val', root=root_path),
                    'test': FinancialGraph(dataset_url=self.cfg.dataset.url,
                                           dataset_name=self.cfg.dataset.name,
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
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = torch.tensor([1])               # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)