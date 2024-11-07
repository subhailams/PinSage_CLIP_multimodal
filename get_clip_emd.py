import os
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class PinDataset(Dataset):
    def __init__(self, image_paths):
        """
        Dataset to load images for CLIP embedding generation.
        Args:
            image_paths (list): List of paths to images.
        """
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        return image, img_path


def generate_clip_embeddings(dataset_path, output_dir, batch_size=32, device="cuda"):
    """
    Generates CLIP embeddings for images and saves them as .npy files in batches.
    
    Args:
        dataset_path (str): Path to the directory containing the images.
        output_dir (str): Directory to save the CLIP embeddings.
        batch_size (int): Batch size for processing images.
        device (str): Device to run CLIP model on, "cpu" or "cuda".
    """
    # Load CLIP model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Get list of all image paths
    image_paths = [os.path.join(dataset_path, fname) for fname in os.listdir(dataset_path) if fname.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_paths)} images.")

    # Create Dataset and DataLoader
    dataset = PinDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    os.makedirs(output_dir, exist_ok=True)
    all_embeddings = []

    # Process images in batches
    with torch.no_grad():
        for batch_images, batch_paths in tqdm(dataloader, desc="Generating CLIP embeddings"):
            # Process batch of images with CLIP
            inputs = clip_processor(images=batch_images, return_tensors="pt").to(device)
            image_features = clip_model.get_image_features(**inputs)
            
            # Normalize embeddings
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().numpy()

            # Save embeddings for each image
            for img_path, embedding in zip(batch_paths, image_features):
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                output_file = os.path.join(output_dir, f"{img_name}_clip_embedding.npy")
                np.save(output_file, embedding)
                all_embeddings.append(output_file)

    print(f"CLIP embeddings saved to {output_dir}.")

    return all_embeddings

if __name__ == "__main__":
    dataset_path = "data/pinterest_images"  # Replace with your dataset path
    output_dir = "data/clip_embeddings"  # Replace with your output directory for embeddings
    batch_size = 32  # Adjust batch size according to your memory capacity
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Generate CLIP embeddings and save as .npy files
    generate_clip_embeddings(dataset_path, output_dir, batch_size=batch_size, device=device)
