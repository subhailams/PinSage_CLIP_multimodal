import dask.dataframe as dd
import dgl
import numpy as np
import scipy.sparse as ssp
import torch
import tqdm

# Step 1: Train-Test Split by Timestamp for Recommender Systems
def train_test_split_by_time(df, timestamp_col='timestamp', user_col='user_id'):
    """
    Split the DataFrame for each user into training, validation, and test sets by time.
    Args:
        df (DataFrame): DataFrame with user-item interactions, including timestamps.
        timestamp_col (str): Column name for timestamps.
        user_col (str): Column name for users.
    Returns:
        Arrays: Indices for training, validation, and test sets.
    """
    df["train_mask"] = np.ones(len(df), dtype=bool)
    df["val_mask"] = np.zeros(len(df), dtype=bool)
    df["test_mask"] = np.zeros(len(df), dtype=bool)
    
    # Convert to Dask DataFrame for efficient processing
    df = dd.from_pandas(df, npartitions=10)

    # Function to split data per user
    def split_user_data(user_df):
        user_df = user_df.sort_values(by=timestamp_col)
        if len(user_df) > 1:
            user_df.iloc[-1, -1] = True  # Set last item as test
            user_df.iloc[-1, -3] = False  # Remove from train
        if len(user_df) > 2:
            user_df.iloc[-2, -2] = True  # Set second last as validation
            user_df.iloc[-2, -3] = False  # Remove from train
        return user_df

    # Define metadata for Dask processing
    meta = {
        "user_id": np.int64,
        "item_id": np.int64,
        "rating": np.float32,
        "timestamp": np.int64,
        "train_mask": bool,
        "val_mask": bool,
        "test_mask": bool
    }

    # Apply function across users and convert to pandas
    df = df.groupby(user_col, group_keys=False).apply(split_user_data, meta=meta).compute()
    
    print("Sample user's sorted interactions by time:\n", 
          df[df[user_col] == df[user_col].unique()[0]].sort_values(timestamp_col))
    
    # Return index arrays for each set
    return df["train_mask"].to_numpy().nonzero()[0], df["val_mask"].to_numpy().nonzero()[0], df["test_mask"].to_numpy().nonzero()[0]


# Step 2: Build Training Subgraph from Indices
def build_train_graph(g, train_indices, user_type='user', item_type='item', edge_type='interacts', edge_type_rev='rev_interacts'):
    """
    Builds a subgraph for training using the specified train indices.
    Args:
        g (DGLGraph): Full DGL graph.
        train_indices (Array): Indices for training edges.
        user_type (str): Node type for users.
        item_type (str): Node type for items.
        edge_type (str): Edge type for interactions.
        edge_type_rev (str): Reverse edge type.
    Returns:
        DGLGraph: Training subgraph.
    """
    train_graph = g.edge_subgraph({edge_type: train_indices, edge_type_rev: train_indices}, relabel_nodes=False)
    
    # Copy node features
    for node_type in g.ntypes:
        for feat_name, feat_data in g.nodes[node_type].data.items():
            train_graph.nodes[node_type].data[feat_name] = feat_data
            
    # Copy edge features
    for edge_type in g.etypes:
        for feat_name, feat_data in g.edges[edge_type].data.items():
            train_graph.edges[edge_type].data[feat_name] = feat_data[train_graph.edges[edge_type].data[dgl.EID]]
    
    return train_graph


# Step 3: Build Validation and Test Sparse Matrices
def build_val_test_matrix(g, val_indices, test_indices, user_type='user', item_type='item', edge_type='interacts'):
    """
    Constructs sparse matrices for validation and test sets.
    Args:
        g (DGLGraph): Full DGL graph.
        val_indices (Array): Indices for validation edges.
        test_indices (Array): Indices for test edges.
        user_type (str): Node type for users.
        item_type (str): Node type for items.
        edge_type (str): Edge type for interactions.
    Returns:
        Tuple: Sparse matrices for validation and test sets.
    """
    n_users = g.num_nodes(user_type)
    n_items = g.num_nodes(item_type)
    
    val_src, val_dst = g.find_edges(val_indices, etype=edge_type)
    test_src, test_dst = g.find_edges(test_indices, etype=edge_type)
    
    val_matrix = ssp.coo_matrix((np.ones(len(val_src)), (val_src.numpy(), val_dst.numpy())), shape=(n_users, n_items))
    test_matrix = ssp.coo_matrix((np.ones(len(test_src)), (test_src.numpy(), test_dst.numpy())), shape=(n_users, n_items))
    
    return val_matrix, test_matrix


# Step 4: Linear Normalization
def linear_normalize(values):
    """
    Normalize values to a range of [0, 1].
    Args:
        values (Array): Values to normalize.
    Returns:
        Array: Normalized values.
    """
    return (values - values.min(0, keepdims=True)) / (values.max(0, keepdims=True) - values.min(0, keepdims=True))


# Example Usage:
# Assuming df is your Pinterest data, and g is your DGL graph object with user-item edges

# Prepare data
# train_indices, val_indices, test_indices = train_test_split_by_time(df, 'timestamp', 'user_id')

# Construct training graph
# train_graph = build_train_graph(g, train_indices, 'user', 'item', 'interacts', 'rev_interacts')

# Build validation and test matrices
# val_matrix, test_matrix = build_val_test_matrix(g, val_indices, test_indices, 'user', 'item', 'interacts')

