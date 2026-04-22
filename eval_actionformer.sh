#!/bin/bash

CONFIG="configs/captaincook_egovlp.yaml"
FEAT_FOLDER="./data/egovlp_features"

echo "====================================================="
echo "   GENERATION OF RESULTS (SAVE ONLY MODE)            "
echo "====================================================="

for FOLD in 1 2 3 4 5
do
    echo "-----------------------------------------------------"
    echo "                 PROCESSING FOLD ${FOLD}             "
    echo "-----------------------------------------------------"

    JSON_PATH="./captaincook_actionformer_annotations/combined/recordings_fold${FOLD}.json"
    CKPT_DIR="./ckpt/ego4d/egovlp_recordings_egovlp_fold${FOLD}"

    python eval.py ${CONFIG} ${CKPT_DIR} \
        --backbone egovlp \
        --division_type recordings \
        --feat_folder ${FEAT_FOLDER} \
        --num_frames 16 \
        --stride 16 \
        --videos_type all \
        --json_file ${JSON_PATH} \
        --saveonly

    echo "Results saved for Fold ${FOLD}!"
done

echo "====================================================="
echo "               OPERATION COMPLETED SUCCESSFULLY                   "
echo "====================================================="