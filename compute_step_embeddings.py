import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def create_step_embeddings(csv_path, feat_dir, output_file, score_threshold=0.05, top_k=100, fps=1.876):
    """
    Filters action localization predictions and creates averaged step-level embeddings.
    
    Args:
        csv_path (str): Path to the predictions CSV file.
        feat_dir (str): Directory containing the original .npz EgoVLP features.
        output_file (str): Path where the final .npz dataset will be saved.
        score_threshold (float): Minimum confidence score to keep a prediction.
        top_k (int): Maximum number of top predictions to keep per video.
        fps (float): Frames per second used for feature extraction.
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Load predictions
    df = pd.read_csv(csv_path)
    print(f"Initial predictions: {len(df)}")

    #  Filter by score to remove low-confidence noise
    df = df[df['score'] >= score_threshold]
    print(f"After score filtering (>= {score_threshold}): {len(df)}")

    # Keep Top-K predictions per video to manage sequence length
    # Sort by score first to select the best candidates
    df = df.sort_values(['video_id', 'score'], ascending=[True, False])
    df = df.groupby('video_id').head(top_k)

    # Sort chronologically per video (Crucial for Transformer models)
    df = df.sort_values(['video_id', 'start_time'])
    
    video_groups = df.groupby('video_id')
    video_step_embeddings = {}

    print(f"Processing {len(video_groups)} videos...")
    for video_id, group in tqdm(video_groups):
        feat_path = os.path.join(feat_dir, f"{video_id}.npz")
        
        if not os.path.exists(feat_path):
            continue
            
        # Load EgoVLP features 
        try:
            npz_data = np.load(feat_path)
            video_features = npz_data['features'] if 'features' in npz_data else npz_data['arr_0']
        except Exception:
            continue
            
        total_feats = video_features.shape[0]
        step_embs = []
        
        for _, row in group.iterrows():
            # Convert timestamps to feature indices
            start_idx = int(row['start_time'] * fps)
            end_idx = int(row['end_time'] * fps)
            
            # Boundary clipping
            start_idx = max(0, min(start_idx, total_feats - 1))
            end_idx = max(start_idx + 1, min(end_idx, total_feats))
                
            # Compute temporal average pooling for the segment
            feat_slice = video_features[start_idx:end_idx]
            if len(feat_slice) > 0:
                step_emb = np.mean(feat_slice, axis=0)
                step_embs.append(step_emb)
            
        if step_embs:
            # Store as [Num_Steps, 768]
            video_step_embeddings[video_id] = np.array(step_embs, dtype=np.float32)

    #Save the structured dataset
    np.savez(output_file, **video_step_embeddings)
    print(f"\nSuccessfully created {output_file}")
    print(f"Total videos processed: {len(video_step_embeddings)}")

if __name__ == "__main__":
    create_step_embeddings(
        csv_path="dataset_substep2_predictions.csv",
        feat_dir="./data/egovlp_features",
        output_file="step_embeddings_dataset.npz"
    )