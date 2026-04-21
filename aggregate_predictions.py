import os
import json
import pandas as pd

def extract_predictions_for_substep2():
    
    ckpt_dir = "./ckpt/ego4d"
    all_videos_data = []

    print("=== Inizio Pooling Predizioni dai 5 fold ===")

    for fold in range(1, 6):
        fold_path = os.path.join(ckpt_dir, f"egovlp_recordings_egovlp_fold{fold}")
        
        if not os.path.exists(fold_path):
            print(f"Skipping Fold {fold}: Cartella non trovata in {fold_path}.")
            continue
            
        result_files = [f for f in os.listdir(fold_path) if f.endswith('.json') and 'results' in f]
        
        if not result_files:
            print(f"Skipping Fold {fold}: Nessun file dei risultati trovato.")
            continue
            
        res_file_path = os.path.join(fold_path, result_files[0])
        print(f"Estraendo dal Fold {fold}: {result_files[0]}")
        
        with open(res_file_path, 'r') as f:
            data = json.load(f)
            
        predictions = data.get('results', data.get('database', data))
        
        for video_id, preds in predictions.items():
            for p in preds:
                all_videos_data.append({
                    'video_id': video_id,
                    'label_id': p.get('label_id', p.get('label', -1)),
                    'score': p.get('score', 1.0),
                    'start_time': p.get('segment', [0,0])[0],
                    'end_time': p.get('segment', [0,0])[1],
                    'fold': fold
                })

    df = pd.DataFrame(all_videos_data)
    
    if df.empty:
        print("\nERRORE: Nessuna predizione estratta.")
        return

    df = df.sort_values(by=['video_id', 'start_time'])
    
    output_csv = "dataset_substep2_predictions.csv"
    df.to_csv(output_csv, index=False)
    
    print("\n=== POOLING COMPLETATO ===")
    print(f"Salvate {len(df)} azioni in '{output_csv}'")
    print(f"Video processati: {df['video_id'].nunique()} / 384")

if __name__ == "__main__":
    extract_predictions_for_substep2()