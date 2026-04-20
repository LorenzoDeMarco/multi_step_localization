#!/usr/bin/env bash

# Export paths and variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONUNBUFFERED=1

# Define variables for Substep 1
BACKBONE="egovlp"
FEAT_FOLDER="./data/egovlp_features"
NUM_FRAMES=90
RUN_NAME="egovlp_baseline_run"

echo "Starting ActionFormer Training with ${BACKBONE} features..."

python train.py configs/captaincook_egovlp.yaml \
    --backbone ${BACKBONE} \
    --division_type recordings \
    --feat_folder ${FEAT_FOLDER} \
    --num_frames ${NUM_FRAMES} \
    --output ${RUN_NAME}

echo "Training completed. Starting Evaluation..."

python eval.py configs/captaincook_egovlp.yaml ${RUN_NAME} \
    --backbone ${BACKBONE} \
    --division_type recordings \
    --feat_folder ${FEAT_FOLDER} \
    --num_frames ${NUM_FRAMES} \
    --videos_type all
    
echo "Evaluation completed. Parsing results..."

# Parse the results into the model_outputs folder
python parse_results.py \
    --results_path ./ckpt/${RUN_NAME}/results_all.json \
    --output_csv ./model_outputs/egovlp_predictions.csv