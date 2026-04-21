
import json
import random
import copy
import os

def create_kfold_splits(json_path, num_folds=5, seed=42):
    print(f"Loading original annotations from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract all video IDs
    video_ids = list(data['database'].keys())
    print(f"Total videos found: {len(video_ids)}")

    # Shuffle for random but reproducible distribution
    random.seed(seed)
    random.shuffle(video_ids)

    # Split into 5 folds
    fold_size = len(video_ids) // num_folds
    folds = []
    for i in range(num_folds):
        start_idx = i * fold_size
        # The last fold catches any remaining elements
        end_idx = (i + 1) * fold_size if i < num_folds - 1 else len(video_ids)
        folds.append(video_ids[start_idx:end_idx])

    # Generate the JSON file for each fold
    output_dir = os.path.dirname(json_path)
    if not output_dir:
        output_dir = "."
    base_name = os.path.basename(json_path).replace('.json', '')

    for i in range(num_folds):
        fold_num = i + 1
        val_ids = set(folds[i])
        
        # All other folds become training data
        train_ids = set([vid for j, f in enumerate(folds) if j != i for vid in f])

        # Deep copy to avoid modifying the original dict references
        fold_data = copy.deepcopy(data)

        # Update the 'subset' field for every video
        for vid, video_info in fold_data['database'].items():
            if vid in train_ids:
                video_info['subset'] = 'training'
            elif vid in val_ids:
                video_info['subset'] = 'validation'

        output_file = os.path.join(output_dir, f"{base_name}_fold{fold_num}.json")
        with open(output_file, 'w') as f:
            json.dump(fold_data, f, indent=4)

        print(f"Created {output_file} -> Training: {len(train_ids)} | Validation: {len(val_ids)}")

input_json_path = "./captaincook_actionformer_annotations/combined/recordings.json"

create_kfold_splits(input_json_path)