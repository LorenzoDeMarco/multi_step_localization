import os
import pickle
import pandas as pd

def extract_predictions_from_pickle():
    # Percorso base su Colab
    base_dir = "/content/code/multi_step_localization/ckpt/ego4d"
    all_videos_data = []

    print("=== Inizio Pooling dai file Pickle ===")

    for fold in range(1, 6):
        fold_folder = f"egovlp_recordings_egovlp_fold{fold}"
        # Il file salvato da eval.py con --saveonly
        file_path = os.path.join(base_dir, fold_folder, "eval_results.pkl")
        
        if not os.path.exists(file_path):
            print(f"File non trovato per Fold {fold}: {file_path}")
            continue
            
        print(f"Caricamento Fold {fold}...")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        # In ActionFormer, il pickle contiene un dizionario dove le chiavi sono i video_id
        # e i valori sono liste di predizioni
        results = data.get('results', data)
        
        for video_id, preds in results.items():
            for p in preds:
                all_videos_data.append({
                    'video_id': video_id,
                    'label_id': p['label'],
                    'score': p['score'],
                    'start_time': p['segment'][0],
                    'end_time': p['segment'][1],
                    'fold': fold
                })

    df = pd.DataFrame(all_videos_data)
    if not df.empty:
        # Ordiniamo per video e tempo per coerenza
        df = df.sort_values(by=['video_id', 'start_time'])
        output_name = "dataset_substep2_predictions.csv"
        df.to_csv(output_name, index=False)
        print(f"\nSuccesso! Creato '{output_name}' con {len(df)} predizioni totali.")
    else:
        print("\nErrore: Nessun dato trovato nei file pkl. Verifica il contenuto dei file.")

if __name__ == "__main__":
    extract_predictions_from_pickle()