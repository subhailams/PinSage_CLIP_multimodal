import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from builder import PandasGraphBuilder
from data_utils import *

import dgl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="Directory containing Pinterest dataset files")
    parser.add_argument("out_directory", type=str, help="Directory to output the processed graph and data")
    args = parser.parse_args()
    directory = args.directory
    out_directory = args.out_directory
    os.makedirs(out_directory, exist_ok=True)

    ## Build heterogeneous graph

    # Load board-pin-category data from board_pin_category_imgid_dataset.csv
    item_metadata = pd.read_csv(os.path.join(directory, "board_pin_category_imgid_dataset.csv"))

    # Ensure item_metadata contains the required columns
    assert 'board_id' in item_metadata.columns and 'pin_id' in item_metadata.columns and 'category_name' in item_metadata.columns, \
        "board_pin_category_imgid_dataset.csv must contain 'board_id', 'pin_id', and 'category_name' columns."

    # Rename columns for consistency with graph terminology
    item_metadata = item_metadata.rename(columns={'board_id': 'user_id', 'pin_id': 'item_id'})

    # Create interactions DataFrame
    interactions = item_metadata[['user_id', 'item_id']].copy()
    interactions['rating'] = 1  # Placeholder rating, as we only need the interaction
    interactions['timestamp'] = 0  # Placeholder timestamp, as timestamp isn't provided

    # Get unique users and items from interactions
    users = interactions["user_id"].unique()
    items = interactions["item_id"].unique()

    # Build user and item dataframes
    users = pd.DataFrame(users, columns=["user_id"]).astype("category")
    items = pd.DataFrame(items, columns=["item_id"]).astype("category")

    # Build graph
    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(users, "user_id", "user")
    graph_builder.add_entities(items, "item_id", "item")
    graph_builder.add_binary_relations(
        interactions, "user_id", "item_id", "interacts"
    )
    graph_builder.add_binary_relations(
        interactions, "item_id", "user_id", "interacted-by"
    )

    g = graph_builder.build()

    # Assign features (none in this case, as we only have user and item IDs)
    g.nodes["user"].data["id"] = torch.LongTensor(users["user_id"].cat.codes.values)
    g.nodes["item"].data["id"] = torch.LongTensor(items["item_id"].cat.codes.values)

    # Add rating and timestamp data as edge features
    for edge_type in ["interacts", "interacted-by"]:
        for data_type in ["rating", "timestamp"]:
            if data_type in interactions.columns:
                g.edges[edge_type].data[data_type] = torch.LongTensor(interactions[data_type].values)

    # Train-validation-test split
    train_indices, val_indices, test_indices = train_test_split_by_time(
        interactions, "timestamp", "user_id"
    )

    # Build the graph with training interactions only
    train_g = build_train_graph(
        g, train_indices, "user", "item", "interacts", "interacted-by"
    )
    assert train_g.out_degrees(etype="interacts").min() > 0

    # Build user-item sparse matrices for validation and test set
    val_matrix, test_matrix = build_val_test_matrix(
        g, val_indices, test_indices, "user", "item", "interacts"
    )

    ## Create item-category mapping
    item_category_mapping = dict(zip(item_metadata['item_id'], item_metadata['category_name']))

    # Create a list of category names corresponding to each item in the graph
    item_categories = [item_category_mapping.get(item, "Unknown") for item in items["item_id"]]

    ## Dump the graph and the datasets

    # Save train graph
    dgl.save_graphs(os.path.join(out_directory, "train_g.bin"), train_g)

    dataset = {
        "val-matrix": val_matrix,
        "test-matrix": test_matrix,
        "item-texts": {
            "category_name": item_categories
        },  # Using category_name instead of item text data
        "item-images": None,  # Placeholder for item image data
        "user-type": "user",
        "item-type": "item",
        "user-to-item-type": "interacts",
        "item-to-user-type": "interacted-by",
        "timestamp-edge-column": "timestamp",
    }

    with open(os.path.join(out_directory, "data.pkl"), "wb") as f:
        pickle.dump(dataset, f)
