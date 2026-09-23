#!/bin/bash
# SGE options
#$ -N random_array_train
#$ -cwd
#$ -l h_rt=24:00:00
#$ -l h_rss=8G
#$ -pe sharedmem 4
#$ -t 1-1500
#$ -m ae
#$ -o output_files/$JOB_NAME.$JOB_ID.$TASK_ID.out
#$ -e output_files/$JOB_NAME.$JOB_ID.$TASK_ID.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SGE_O_WORKDIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/output_files"

. /etc/profile.d/modules.sh
module load anaconda
eval "$(conda shell.bash hook)"
conda activate /exports/eddie/scratch/s1948145/anaconda/envs/chem

OUT_DIR="${MODEL_TRAIN_RANDOM_OUT_DIR:-/exports/eddie/scratch/s1948145/Model_ckpts_random_rerun}"
SEEDS=(0 1 2 3 4)
SPLITS=("random_seed0" "random_seed1" "random_seed2" "random_seed3" "random_seed4")
MODELS=("rf" "mlp" "linear" "xgboost")
DATASETS=("HOF" "CD" "IS")
LIST_FILE="${PROJECT_DIR}/03_SH_SCRIPTS/embeddings.txt"

N_ENCODERS=$(wc -l < "${LIST_FILE}")
N_DATASETS=${#DATASETS[@]}
N_MODELS=${#MODELS[@]}
N_SPLITS=${#SPLITS[@]}
N_SEEDS=${#SEEDS[@]}
TOTAL_TASKS=$((N_ENCODERS * N_DATASETS * N_MODELS * N_SPLITS * N_SEEDS))

if (( SGE_TASK_ID < 1 || SGE_TASK_ID > TOTAL_TASKS )); then
  echo "SGE_TASK_ID=${SGE_TASK_ID} is out of range [1, ${TOTAL_TASKS}]. Exiting."
  exit 0
fi

export OMP_NUM_THREADS="${NSLOTS:-1}"
export MKL_NUM_THREADS="${NSLOTS:-1}"
export OPENBLAS_NUM_THREADS="${NSLOTS:-1}"
export NUMEXPR_NUM_THREADS="${NSLOTS:-1}"

idx=$((SGE_TASK_ID - 1))
enc_idx=$((idx % N_ENCODERS)); idx=$((idx / N_ENCODERS))
dataset_idx=$((idx % N_DATASETS)); idx=$((idx / N_DATASETS))
model_idx=$((idx % N_MODELS)); idx=$((idx / N_MODELS))
split_idx=$((idx % N_SPLITS)); idx=$((idx / N_SPLITS))
seed_idx=$((idx % N_SEEDS))

ENCODER="$(sed -n "$((enc_idx + 1))p" "${LIST_FILE}")"
DATASET="${DATASETS[$dataset_idx]}"
MODEL="${MODELS[$model_idx]}"
SPLIT="${SPLITS[$split_idx]}"
SEED="${SEEDS[$seed_idx]}"

echo "JOB_ID=${JOB_ID} TASK_ID=${SGE_TASK_ID} HOST=$(hostname) NSLOTS=${NSLOTS:-1}"
echo "Running: dataset=${DATASET} split=${SPLIT} model=${MODEL} encoder=${ENCODER} seed=${SEED}"

python -u 05_PY_SCRIPTS/run_model_train.py \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --embedding "${ENCODER}" \
  --model "${MODEL}" \
  --output_dir "${OUT_DIR}" \
  --seed "${SEED}"
