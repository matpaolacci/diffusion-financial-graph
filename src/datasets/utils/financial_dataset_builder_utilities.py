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
        Returns a DataFrame containing all accounts in the input DataFrame, along with their laundering status.

        Parameters:
            df (pd.DataFrame): DataFrame containing transaction data with columns 'Source Account', 'Destination Account', and 'Is Laundering'.

        Returns:
            pd.DataFrame: DataFrame with columns ['Account', 'Is Laundering'] listing all unique accounts and their laundering status.
        """

        # Get all illicit transactions
        suspicious = df[df['Is Laundering']==1]

        # Separate source and destination accounts involved in illicit transactions.
        source_df = suspicious[['Source Account', 'Is Laundering']].rename({'Source Account': 'Account'}, axis=1)
        destination_df = suspicious[['Destination Account', 'Is Laundering']].rename({'Destination Account': 'Account'}, axis=1)

        # Joint into a single DataFrame
        suspicious = pd.concat([source_df, destination_df], join='outer')

        # An account could be involved in several transactions, so we drop duplicates
        suspicious = suspicious.drop_duplicates()

        # Get all unique accounts from source and destination
        source_accounts = df[['Source Account']].rename({'Source Account': 'Account'}, axis=1)
        dest_accounts = df[['Destination Account']].rename({'Destination Account': 'Account'}, axis=1)
        all_accounts = pd.concat([source_accounts, dest_accounts], join='outer')
        all_accounts = all_accounts.drop_duplicates()

        all_accounts['Is Laundering'] = 0

        # Mark all the transactions of the accounts involved in illicit transactions as illicit
        all_accounts.set_index('Account', inplace=True)
        all_accounts.update(suspicious.set_index('Account'))
        return all_accounts.reset_index()

    @staticmethod
    def preprocess(df: pd.DataFrame, csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        # The dataframe now only contains: Source Account, Destination Account, Is Laundering, count, affinity
        # No need for complex preprocessing since we already simplified the data
        edges_df = df.sort_values(by=['Source Account'])

        return (
            edges_df,
            DatasetBuilderUtilities._get_nodes(edges_df)
        )