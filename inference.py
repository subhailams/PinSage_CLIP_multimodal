import torch
import dgl
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pickle
import argparse
from model import PinSAGEModel

def load_model_checkpoint(model, checkpoint_path):
    """Load the saved model checkpoint."""
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    print(f"Model loaded from {checkpoint_path}")

@torch.no_grad()
def infer_item_representations(model, dataloader, device):
    """Generate item representations for all items using the model."""
    h_item_batches = []
    for blocks in tqdm(dataloader, desc="Generating item representations"):
        for i in range(len(blocks)):
            blocks[i] = blocks[i].to(device)
        h_item_batches.append(model.get_repr(blocks))
    h_item = torch.cat(h_item_batches, 0)
    return h_item

def recommend_items(model, user_ids, h_item, top_k=10):
    """Recommend top_k items for each user based on item representations."""
    recommendations = {}
    for user_id in user_ids:
        user_representation = model.scorer.get_user_representation(user_id)
        scores = torch.matmul(h_item, user_representation.T)
        top_k_indices = torch.topk(scores, top_k).indices
        recommendations[user_id.item()] = top_k_indices.cpu().numpy()
    return recommendations

def calculate_hit_ratio(recommendations, val_matrix, top_k=10):
    """Calculate hit ratio using the validation matrix."""
    hits = 0
    total_users = len(recommendations)
    for user_id, recommended_items in recommendations.items():
        ground_truth_items = val_matrix[user_id].nonzero()[1]  # Get non-zero columns for the user
        hits += any(item in ground_truth_items for item in recommended_items)
    hit_ratio = hits / total_users
    return hit_ratio

def main_inference(args):
    # Load dataset
    data_info_path = os.path.join(args.dataset_path, "data.pkl")
    with open(data_info_path, "rb") as f:
        dataset = pickle.load(f)
    print(dataset.keys())

    # Use val-matrix for evaluation
    val_matrix = dataset["val-matrix"].tocsr()
    item_ntype = dataset["item-type"]

    device = torch.device(args.device)

    # Create a minimal graph to pass to the model
    minimal_graph = dgl.heterograph({
        (item_ntype, 'dummy-relation', item_ntype): ([0], [0])  # Minimal self-loop edge
    })

# Verify and unpack textset fields correctly
    for key, field in dataset["item-texts"].items():
        if len(field) == 4:
            textlist, vocab, pad_var, batch_first = field
        else:
            print(f"Unexpected field structure for key '{key}': {field}")
            continue
            
    # Load the trained model
    model = PinSAGEModel(
        minimal_graph,
        item_ntype,
        dataset["item-texts"],
        args.hidden_dims,
        args.num_layers
    ).to(device)
    load_model_checkpoint(model, args.checkpoint_path)


   

    # Prepare the DataLoader for item representations
    collator = sampler_module.PinSAGECollator(None, None, item_ntype, dataset["item-texts"])
    dataloader_test = DataLoader(
        torch.arange(val_matrix.shape[1]),  # Use the number of items in val_matrix
        batch_size=args.batch_size,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers,
    )

    # Generate item representations
    h_item = infer_item_representations(model, dataloader_test, device)

    # Recommend items to new users (Example: user IDs 0 to 10)
    user_ids = torch.arange(0, 10)
    recommendations = recommend_items(model, user_ids, h_item, top_k=args.k)

    # Calculate and print hit ratio
    hit_ratio = calculate_hit_ratio(recommendations, val_matrix, top_k=args.k)
    print(f"Hit Ratio: {hit_ratio:.4f}")

    # Print recommendations
    for user_id, recommended_items in recommendations.items():
        print(f"User {user_id}: Recommended items {recommended_items}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--hidden-dims", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("-k", type=int, default=10, help="Top-k items to recommend")
    args = parser.parse_args()

    main_inference(args)
