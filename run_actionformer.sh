#!/bin/bash

CONFIG="configs/captaincook_egovlp.yaml"
FEAT_FOLDER="./data/egovlp_features"
NUM_FRAMES=16
STRIDE=16

echo "====================================================="
echo "   STARTING 5-FOLD CROSS-VALIDATION ACTIONFORMER     "
echo "====================================================="

for FOLD in 1
do
    echo "-----------------------------------------------------"
    echo "                 INIZIO FOLD ${FOLD}                 "
    echo "-----------------------------------------------------"
    
    JSON_PATH="./captaincook_actionformer_annotations/combined/recordings_fold${FOLD}.json"
    RUN_NAME="egovlp_fold${FOLD}"
    
    #TRAINING
    echo "-> Training Fold ${FOLD}..."
    python train.py ${CONFIG} \
        --backbone egovlp \
        --division_type recordings \
        --feat_folder ${FEAT_FOLDER} \
        --num_frames ${NUM_FRAMES} \
        --stride ${STRIDE} \
        --json_file ${JSON_PATH} \
        --output ${RUN_NAME}
        
    #EVALUATION
    echo "-> Evaluation Fold ${FOLD}..."
    python eval.py ${CONFIG} ${RUN_NAME} \
        --videos_type all \
        --json_file ${JSON_PATH}
        
    echo "Fold ${FOLD} completato!"
    echo "-----------------------------------------------------"
done

echo "====================================================="
echo "   5-FOLD CROSS-VALIDATION COMPLETATA CON SUCCESSO!  "
echo "====================================================="