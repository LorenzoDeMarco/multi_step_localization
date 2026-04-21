import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def create_step_embeddings():
    csv_path = "pred_segments_error_dataset.csv"  
    feat_dir = "./data/egovlp_features"            
    output_file = "step_embeddings_dataset.npz"
    
    fps = 1.876 
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        return

    print(f"Loading predictions from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    rename_dict = {
        'video-id': 'video_id',
        't-start': 'start_time',
        't-end': 'end_time'
    }
    df = df.rename(columns=rename_dict)
    
    if 'type' in df.columns:
        df = df[df['type'] == 'pred']
    
    video_groups = df.groupby('video_id')
    video_step_embeddings = {}
    
    print(f"Computing step-level embeddings for {len(video_groups)} videos...")
    for video_id, group in tqdm(video_groups):
        feat_path = os.path.join(feat_dir, f"{video_id}.npz")
        
        if not os.path.exists(feat_path):
            continue
            
        try:
            npz_data = np.load(feat_path)
            video_features = npz_data['features'] if 'features' in npz_data else npz_data['arr_0']
        except Exception:
            continue
            
        total_feats = video_features.shape[0]
        step_embs = []
        
        for _, row in group.iterrows():
            start_idx = int(row['start_time'] * fps)
            end_idx = int(row['end_time'] * fps)
            
            # Clipping boundaries
            start_idx = max(0, min(start_idx, total_feats - 1))
            end_idx = max(start_idx + 1, min(end_idx, total_feats))
                
            # Average Pooling
            feat_slice = video_features[start_idx:end_idx]
            if len(feat_slice) > 0:
                step_emb = np.mean(feat_slice, axis=0)
                step_embs.append(step_emb)
            
        if step_embs:
            video_step_embeddings[video_id] = np.array(step_embs, dtype=np.float32)

    # Save as .npz
    np.savez(output_file, **video_step_embeddings)
    print(f"\nProcessing complete!")
    print(f"Final dataset saved in: {output_file}")
    print(f"Videos in dataset: {len(video_step_embeddings)}")

if __name__ == "__main__":
    create_step_embeddings()