"""
Module for computing step-level embeddings from frame-level features
using ActionFormer temporal predictions.
"""

import numpy as np
import pandas as pd
import os
import argparse

def process_video_embeddings(video_id, predictions_df, features_dir, feature_fps=1.0):
    video_preds = predictions_df[predictions_df['video_id'] == video_id]
    feature_path = os.path.join(features_dir, f"{video_id}.npy")
    
    if not os.path.exists(feature_path):
        print(f"Warning: Features for {video_id} not found at {feature_path}. Skipping.")
        return []
        
    video_features = np.load(feature_path)
    step_embeddings = []
    
    for _, row in video_preds.iterrows():
        start_idx = int(np.floor(row['start'] * feature_fps))
        end_idx = int(np.ceil(row['end'] * feature_fps))
        
        # Boundary enforcement
        if start_idx >= end_idx:
            end_idx = start_idx + 1
            
        end_idx = min(end_idx, len(video_features))
        start_idx = min(start_idx, len(video_features) - 1)
        
        # Average pooling for the single step
        segment_features = video_features[start_idx:end_idx, :]
        step_emb = np.mean(segment_features, axis=0) 
        
        step_embeddings.append({
            'video_id': video_id,
            'start': row['start'],
            'end': row['end'],
            'label': row['label'],
            'embedding': step_emb
        })
        
    return step_embeddings

def main():
    parser = argparse.ArgumentParser(description="Compute step-level embeddings.")
    parser.add_argument('--preds_csv', type=str, required=True, help="Path to parsed predictions CSV")
    parser.add_argument('--feat_dir', type=str, required=True, help="Directory containing .npy features")
    parser.add_argument('--output_dir', type=str, default="step_embeddings", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    preds_df = pd.read_csv(args.preds_csv)
    unique_videos = preds_df['video_id'].unique()
    
    for vid in unique_videos:
        embeddings = process_video_embeddings(vid, preds_df, args.feat_dir)
        if embeddings:
            # Save the computed step-level embeddings for each video
            output_file = os.path.join(args.output_dir, f"{vid}_step_embeddings.npy")
            np.save(output_file, embeddings)
            print(f"Saved {len(embeddings)} step embeddings for video {vid}")

if __name__ == "__main__":
    main()