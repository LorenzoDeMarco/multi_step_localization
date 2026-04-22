#!/bin/bash

CONFIG="configs/captaincook_egovlp.yaml"
FEAT_FOLDER="./data/egovlp_features"
NUM_FRAMES=16
STRIDE=16

echo "====================================================="
echo "   STARTING 5-FOLD CROSS-VALIDATION ACTIONFORMER     "
echo "====================================================="

for FOLD in 1 2 3 4 5
do
    echo "-----------------------------------------------------"
    echo "                 START FOLD ${FOLD}                 "
    echo "-----------------------------------------------------"
    
    JSON_PATH="./captaincook_actionformer_annotations/combined/recordings_fold${FOLD}.json"
    
    # TRAINING
    echo "-> Training Fold ${FOLD}..."
    python train.py ${CONFIG} \
        --backbone egovlp \
        --division_type recordings \
        --feat_folder ${FEAT_FOLDER} \
        --num_frames ${NUM_FRAMES} \
        --stride ${STRIDE} \
        --json_file ${JSON_PATH} \
        --output fold${FOLD}
        
    echo "Fold ${FOLD} completed successfully!"
    echo "-----------------------------------------------------"
done

echo "====================================================="
echo "               ALL FOLDS COMPLETED SUCCESSFULLY       "
echo "====================================================="