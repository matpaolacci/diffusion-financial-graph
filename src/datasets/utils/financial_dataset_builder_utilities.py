import pandas as pd
from graph_tool.all import label_components, Graph, GraphView

class DatasetBuilderUtilities:
    @staticmethod
    def _get_largest_component_df(edges_df: pd.DataFrame, csv_path: str) -> pd.DataFrame:
        """
        Extract the largest component of the graph.
        :param edges_df: The dataframe with the edges of the graph.
        :return: A DataFrame containing only the edges from the largest component.
        """
        print(f'Processing df from [{csv_path}], getting the largest component...')
        
        trans_graph = Graph(
            list(edges_df[['Source Account', 'Destination Account']].itertuples(index=False, name=None)),
            hashed=True,
            directed=False
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
        """
        Returns a DataFrame containing all accounts in the input DataFrame, along with their associated banks and laundering status.

        Parameters:
            df (pd.DataFrame): DataFrame containing transaction data with columns such as 'Source Account', 'Destination Account', 'From Bank', 'To Bank', and 'Is Laundering'.

        Returns:
            pd.DataFrame: DataFrame with columns ['Account', 'Bank', 'Is Laundering'] listing all unique accounts, their banks, and laundering status.
        """

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
    def preprocess(df: pd.DataFrame, csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        :return:
            laundering_edges_df: A DataFrame containing only the edges involved in laundering transactions.
            laundering_accounts: A DataFrame containing only the accounts involved in laundering transactions.
        """
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

        return (
            largest_comp_edges_df[largest_comp_edges_df['Is Laundering'] == 1],
            DatasetBuilderUtilities._get_nodes(
                largest_comp_edges_df[largest_comp_edges_df['Is Laundering'] == 1]
            )
        )
    
    @staticmethod
    def get_undirected_edges_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add the reverse edges to the dataframe in such a way as to make the graph represented by the dataframe undirected.
        
        Parameters:
        df (pd.DataFrame): DataFrame with columns ['Source Account', 'Destination Account', ...]
                        representing edges and their features
        
        Returns:
        pd.DataFrame: Symmetric dataframe with both (i,j) and (j,i) edges
        """
        
        # Create a set of existing edges for fast lookup
        existing_edges = set(zip(df['Source Account'], df['Destination Account']))
        
        # Find edges that need their reverse added
        missing_reverse_edges = []
        
        for _, row in df.iterrows():
            src, dst, src_bank, dst_bank = \
                row['Source Account'], row['Destination Account'], row['From Bank'], row['To Bank']
            
            # Check if reverse edge (dst, src) exists
            if (dst, src) not in existing_edges:
                # Create reverse edge with same features
                reverse_row = row.copy()
                reverse_row['Source Account'] = dst
                reverse_row['Destination Account'] = src
                reverse_row['From Bank'] = dst_bank
                reverse_row['To Bank'] = src_bank
                missing_reverse_edges.append(reverse_row)
        
        # Add missing reverse edges to original dataframe
        if missing_reverse_edges:
            missing_df = pd.DataFrame(missing_reverse_edges)
            symmetric_df = pd.concat([df, missing_df], ignore_index=True)
        else:
            symmetric_df = df.copy()
        
        return symmetric_df.reset_index(drop=True)
