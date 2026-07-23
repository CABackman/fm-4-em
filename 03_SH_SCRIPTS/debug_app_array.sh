#!/bin/bash
# SGE options
#$ -N APP_rf_array
#$ -cwd
#$ -l h_rt=08:00:00
#$ -l h_rss=16G
#$ -pe sharedmem 4
#$ -m ae
#$ -M s1948145@ed.ac.uk
#$ -o output_files/$JOB_NAME.$JOB_ID.out
#$ -e output_files/$JOB_NAME.$JOB_ID.err

set -euo pipefail 

echo "Running debug job"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
python --version

. /etc/profile.d/modules.sh
module load anaconda/2024.02
source activate /exports/eddie/scratch/s1948145/anaconda/envs/chem


OUT_DIR="/exports/eddie/scratch/s1948145/Model_ckpts"
LIST_FILE="03_SH_SCRIPTS/embeddings.txt"
DATASETS=("HOF_exp")
MODELS=("tabfm")

# Pick the embedding for this task
emb_name="molclr"

if [[ -z "${emb_name}" ]]; then
  echo "No embedding found for SGE_TASK_ID=$SGE_TASK_ID (check $LIST_FILE)"
  exit 1
fi

echo "Embedding: $emb_name"

# Loop over datasets and models
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do

    echo "Running: dataset=$dataset model=$model embedding=$emb_name"

    python 05_PY_SCRIPTS/run_model_train.py \
      --dataset "$dataset" \
      --embedding "$emb_name" \
      --model "$model" \
      --output_dir "$OUT_DIR" \
      --split "random_seed0"

  done
done