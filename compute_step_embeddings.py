import pandas as pd
import numpy as np
import os
import torch
from tqdm import tqdm

def create_step_embeddings():
    # File paths
    csv_path = "actionformer_predictions_full.csv"  # The CSV generated from ActionFormer
    feat_dir = "./data/egovlp_features"            # Folder with original .npz files
    output_file = "step_embeddings_dataset.pt"
    
    fps = 1.876 
    
    print(f"Loading predictions from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Group predictions by video
    video_groups = df.groupby('video_id')
    
    # Dictionary to store the final tensors
    # Format: { "video_id": Tensor of shape [num_steps, 768] }
    video_step_embeddings = {}
    
    print("Computing step-level embeddings...")
    for video_id, group in tqdm(video_groups):
        feat_path = os.path.join(feat_dir, f"{video_id}.npz")
        
        if not os.path.exists(feat_path):
            print(f"Warning: Feature file missing for {video_id}")
            continue
            
        # Load the original EgoVLP features for this video
        try:
            npz_data = np.load(feat_path)
            video_features = npz_data['features'] if 'features' in npz_data else npz_data['arr_0']
        except Exception as e:
            print(f"Error loading {feat_path}: {e}")
            continue
            
        total_feats = video_features.shape[0]
        step_embs = []
        
        # Iterate over each predicted action (step) in the video
        for _, row in group.iterrows():
            start_time = row['start_time']
            end_time = row['end_time']
            
            # Convert timestamps (seconds) to feature array indices
            start_idx = int(start_time * fps)
            end_idx = int(end_time * fps)
            
            # Handle edge cases
            if start_idx >= total_feats:
                start_idx = total_feats - 1
            if end_idx <= start_idx:
                end_idx = start_idx + 1 # Ensure at least 1 feature is selected
            if end_idx > total_feats:
                end_idx = total_feats
                
            # Extract all features within the (start, end) boundaries
            feat_slice = video_features[start_idx:end_idx]
            
            # Compute a unique step-level embedding by averaging
            step_emb = np.mean(feat_slice, axis=0)
            
            step_embs.append(step_emb)
            
        # Convert the list of embeddings into a PyTorch tensor
        # Final shape for this video: [Number_of_steps, 768]
        video_tensor = torch.tensor(np.array(step_embs), dtype=torch.float32)
        video_step_embeddings[video_id] = video_tensor

    torch.save(video_step_embeddings, output_file)
    print(f"\nSuccessfully saved step embeddings to {output_file}")
    print(f"Total videos processed: {len(video_step_embeddings)}")

if __name__ == "__main__":
    create_step_embeddings()