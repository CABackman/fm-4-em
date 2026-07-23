#!/bin/bash

# Script to test 05_run_model_train.py locally using an embedding from 02_EMBEDDINGS
# Output goes to a temp dir inside 07_outputs

set -e

# Path to embedding (pick the first .npy file found)
EMBEDDING="hq_descriptor"
SPLIT=""cluster_label0""

# Create temp output directory
OUTDIR="07_OUTPUTS/temp/"
mkdir -p "$OUTDIR"

# Run the model training script
python 05_PY_SCRIPTS/run_model_train.py \
  --embedding "$EMBEDDING" \
  --split "$SPLIT" \
  --output_dir "$OUTDIR" \
  --dataset "HOF" \
  --model "linear" \

# Run the model training script
python 05_PY_SCRIPTS/run_model_train.py \
  --embedding "$EMBEDDING" \
  --split "$SPLIT" \
  --output_dir "$OUTDIR" \
  --dataset "HOF" \
  --model "xgboost" \

  # Run the model training script
python 05_PY_SCRIPTS/run_model_train.py \
  --embedding "$EMBEDDING" \
  --split "$SPLIT" \
  --output_dir "$OUTDIR" \
  --dataset "HOF" \
  --model "mlp" \
  
# Run the model training script
python 05_PY_SCRIPTS/run_model_train.py \
  --embedding "$EMBEDDING" \
  --split "$SPLIT" \
  --output_dir "$OUTDIR" \
  --dataset "HOF" \
  --model "rf" \


echo "Training completed. Output written to $OUTDIR"