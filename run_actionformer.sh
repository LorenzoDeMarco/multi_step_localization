#!/usr/bin/env bash

# Export paths and variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONUNBUFFERED=1

# Define variables for Substep 1
BACKBONE="egovlp"
FEAT_FOLDER="./data/egovlp_features"
NUM_FRAMES=16
STRIDE=16
RUN_NAME="egovlp_baseline_run"

echo "Starting ActionFormer Training with ${BACKBONE} features..."

python train.py configs/captaincook_egovlp.yaml \
    --backbone ${BACKBONE} \
    --division_type recordings \
    --feat_folder ${FEAT_FOLDER} \
    --num_frames ${NUM_FRAMES} \
    --stride ${STRIDE} \
    --output ${RUN_NAME}

echo "Training completed. Starting Evaluation..."

python eval.py configs/captaincook_egovlp.yaml ${RUN_NAME} \
    --backbone ${BACKBONE} \
    --division_type recordings \
    --feat_folder ${FEAT_FOLDER} \
    --num_frames ${NUM_FRAMES} \
    --stride ${STRIDE} \
    --videos_type all
    
echo "Evaluation completed. Parsing results..."

# Parse the results into the model_outputs folder
python parse_results.py