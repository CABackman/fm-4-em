#!/bin/bash

# Script to test 05_run_model_train.py locally using an embedding from 02_EMBEDDINGS
# Output goes to a temp dir inside 07_outputs

set -e

# Path to embedding (pick the first .npy file found)
EMBEDDING=("chemeleon")
SPLIT=("random_seed0" "random_seed1" "random_seed2" "random_seed3" "random_seed4")
DATASET=("HOF_exp" "log(h50)_exp" "CD")
MODEL="tabfm"

# Create temp output directory
OUTDIR="07_OUTPUTS/20260610/"
mkdir -p "$OUTDIR"

# Run the model training script
DATE="$(date +%Y%m%d)"

for DATASET in "${DATASET[@]}"; do
  for EMBEDDING in "${EMBEDDING[@]}"; do
    for SPLIT in "${SPLIT[@]}"; do
      OUTDIR="07_OUTPUTS/tmp/${DATE}/${DATASET}/${MODEL}/${EMBEDDING}/${SPLIT}/"
      mkdir -p "$OUTDIR"

      python 05_PY_SCRIPTS/run_model_train.py \
        --embedding "$EMBEDDING" \
        --split "$SPLIT" \
        --output_dir "$OUTDIR" \
        --dataset "$DATASET" \
        --model "$MODEL"

      echo "Training completed. Output written to $OUTDIR"
    done
  done
done

echo "Training completed. Output written to $OUTDIR"