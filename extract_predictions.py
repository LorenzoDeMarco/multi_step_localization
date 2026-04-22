import os
import pickle
import pandas as pd
import numpy as np

def extract_predictions():
    base_dir = "./ckpt/ego4d"
    all_data_list = []

    print("=== Extraction of Predictions ===")

    for fold in range(1, 6):
        fold_folder = f"egovlp_recordings_egovlp_fold{fold}"
        file_path = os.path.join(base_dir, fold_folder, "eval_results.pkl")

        if not os.path.exists(file_path):
            continue

        print(f"Loading Fold {fold}...")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        try:
            df_temp = pd.DataFrame(data)

            if df_temp.empty and isinstance(data, dict):
                df_temp = pd.DataFrame.from_dict(data, orient='index').reset_index()

            df_temp.columns = [str(c).replace('-', '_') for c in df_temp.columns]

            rename_map = {'label': 'label_id', 't_start': 'start_time', 't_end': 'end_time', 'video_id': 'video_id'}
            df_temp = df_temp.rename(columns=rename_map)

            if 'video_id' in df_temp.columns:
                df_temp = df_temp[df_temp['video_id'] != 'video-id']

            if 'type' in df_temp.columns:
                df_temp = df_temp[df_temp['type'] == 'pred']

            df_temp['fold'] = fold
            all_data_list.append(df_temp)

        except Exception as e:
            print(f"Errore tecnico nel Fold {fold}: {e}")

    if all_data_list:
        final_df = pd.concat(all_data_list, ignore_index=True)

        for col in ['start_time', 'end_time', 'score']:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

        final_df = final_df.dropna(subset=['start_time', 'end_time'])
        final_df.to_csv("dataset_substep2_predictions.csv", index=False)
        print(f"\nSUCCESS! Created CSV with {len(final_df)} rows.")
    else:
        print("\nError: No data extracted. Please check that the .pkl files are not corrupted.")

if __name__ == "__main__":
    extract_predictions()