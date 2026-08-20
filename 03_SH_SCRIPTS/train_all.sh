#!/bin/bash
# SGE options
#$ -N cluster_array_train
#$ -cwd
#$ -l h_rt=24:00:00
#$ -l h_rss=8G
#$ -pe sharedmem 4
#$ -m ae
#$ -o output_files/$JOB_NAME.$JOB_ID.$TASK_ID.out
#$ -e output_files/$JOB_NAME.$JOB_ID.$TASK_ID.err

set -euo pipefail 

mkdir -p output_files


echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
python --version

if [[ -f /etc/profile.d/modules.sh ]]; then
  . /etc/profile.d/modules.sh
  module load anaconda
fi
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-chem}"


# Output root for results
OUT_DIR="${OUTPUT_DIR:-07_OUTPUTS}"

# repo conventions:
# - Data CSV:     01_DATASETS/<dataset>/*.csv
# - Split indices:01_DATASETS/<dataset>/<split>_split_test_idx.npy
# - Embeddings:   02_EMBEDDINGS/<dataset>/<encoder>.npy

# Define choices
SEEDS=(0 1 2 3 4)
SPLITS=("cluster_label0" "cluster_label1" "cluster_label2" "cluster_label3" "cluster_label4")
MODELS=("rf" "mlp" "linear" "xgboost")
DATASETS=("HOF" "CD" "IS")

# Pick the embedding for this task
LIST_FILE="03_SH_SCRIPTS/embeddings.txt"
N_ENCODERS=$(wc -l < "$LIST_FILE")
N_DATASETS=${#DATASETS[@]}
N_MODELS=${#MODELS[@]}
N_SPLITS=${#SPLITS[@]}
N_SEEDS=${#SEEDS[@]}

TOTAL_TASKS=$((N_ENCODERS * N_DATASETS * N_MODELS * N_SPLITS * N_SEEDS))
if (( SGE_TASK_ID < 1 || SGE_TASK_ID > TOTAL_TASKS )); then
  echo "SGE_TASK_ID=$SGE_TASK_ID is out of range [1, $TOTAL_TASKS]. Exiting."
  exit 0
fi

# Prevent oversubscription (esp. sklearn/numpy):
export OMP_NUM_THREADS="${NSLOTS:-1}"
export MKL_NUM_THREADS="${NSLOTS:-1}"
export OPENBLAS_NUM_THREADS="${NSLOTS:-1}"
export NUMEXPR_NUM_THREADS="${NSLOTS:-1}"

# Map SGE_TASK_ID -> (encoder, dataset, model, split, seed)
idx=$((SGE_TASK_ID - 1))

enc_idx=$(( idx % N_ENCODERS )); idx=$(( idx / N_ENCODERS ))
dataset_idx=$(( idx % N_DATASETS )); idx=$(( idx / N_DATASETS ))
model_idx=$(( idx % N_MODELS )); idx=$(( idx / N_MODELS ))
split_idx=$(( idx % N_SPLITS )); idx=$(( idx / N_SPLITS ))
seed_idx=$(( idx % N_SEEDS )); idx=$(( idx / N_SEEDS ))

ENCODER="$(sed -n "$((enc_idx + 1))p" "$LIST_FILE")"
DATASET="${DATASETS[$dataset_idx]}"
MODEL="${MODELS[$model_idx]}"
SPLIT="${SPLITS[$split_idx]}"
SEED="${SEEDS[$seed_idx]}"


echo "JOB_ID=$JOB_ID TASK_ID=$SGE_TASK_ID HOST=$(hostname) NSLOTS=${NSLOTS:-1}"
echo "Running: dataset=$DATASET split=$SPLIT model=$MODEL encoder=$ENCODER seed=$SEED"

python 05_PY_SCRIPTS/run_model_train.py \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --embedding "$ENCODER" \
  --model "$MODEL" \
  --output_dir "$OUT_DIR" \
  --seed "$SEED"
