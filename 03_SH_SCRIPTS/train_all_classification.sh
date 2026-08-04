#!/bin/bash
# SGE options
#$ -N classification_train
#$ -cwd
#$ -l h_rt=12:00:00
#$ -l h_rss=8G
#$ -pe sharedmem 4
#$ -m ae
#$ -o output_files/$JOB_NAME.$JOB_ID.$TASK_ID.out
#$ -e output_files/$JOB_NAME.$JOB_ID.$TASK_ID.err
#$ -t 1-600

# Submit from the repository root:
#   mkdir -p output_files
#   qsub 03_SH_SCRIPTS/train_all_classification.sh

set -euo pipefail

# SGE -cwd means this should be submitted from the repository root.
export PROJECT_DIR="${PROJECT_DIR:-$PWD}"
OUT_DIR="${CLASSIFICATION_OUT_DIR:-/exports/eddie/scratch/s1948145/Classification_ckpts}"

CLASSIFICATIONS=("binary" "tertiary" "quaternary" "quinary")
EMBEDDINGS=("hq_descriptor" "chemeleon" "morgan_fps" "maceoff_large" "molclr")
BASE_MODELS=("svc" "logreg" "rf" "lightgbm")
ADDITIONAL_MODELS=("bayes" "xgboost")
SEEDS=(0 1 2 3 4)

N_EMBEDDINGS=${#EMBEDDINGS[@]}
N_CLASSIFICATIONS=${#CLASSIFICATIONS[@]}
N_BASE_MODELS=${#BASE_MODELS[@]}
N_ADDITIONAL_MODELS=${#ADDITIONAL_MODELS[@]}
N_SEEDS=${#SEEDS[@]}
BASE_TASKS=$((N_CLASSIFICATIONS * N_EMBEDDINGS * N_BASE_MODELS * N_SEEDS))
ADDITIONAL_TASKS=$((N_CLASSIFICATIONS * N_EMBEDDINGS * N_ADDITIONAL_MODELS * N_SEEDS))
TOTAL_TASKS=$((BASE_TASKS + ADDITIONAL_TASKS))

if (( SGE_TASK_ID < 1 || SGE_TASK_ID > TOTAL_TASKS )); then
  echo "SGE_TASK_ID=$SGE_TASK_ID is out of range [1, $TOTAL_TASKS]. Exiting."
  exit 0
fi

# One parallel CV fit per allocated core; each estimator itself stays single-threaded.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Base models occupy tasks 1-400. The two additional models are tasks 401-600.
if (( SGE_TASK_ID <= BASE_TASKS )); then
  idx=$((SGE_TASK_ID - 1))
  classification_idx=$((idx % N_CLASSIFICATIONS)); idx=$((idx / N_CLASSIFICATIONS))
  embedding_idx=$((idx % N_EMBEDDINGS)); idx=$((idx / N_EMBEDDINGS))
  model_idx=$((idx % N_BASE_MODELS)); idx=$((idx / N_BASE_MODELS))
  seed_idx=$((idx % N_SEEDS))
  MODEL="${BASE_MODELS[$model_idx]}"
else
  idx=$((SGE_TASK_ID - BASE_TASKS - 1))
  classification_idx=$((idx % N_CLASSIFICATIONS)); idx=$((idx / N_CLASSIFICATIONS))
  embedding_idx=$((idx % N_EMBEDDINGS)); idx=$((idx / N_EMBEDDINGS))
  model_idx=$((idx % N_ADDITIONAL_MODELS)); idx=$((idx / N_ADDITIONAL_MODELS))
  seed_idx=$((idx % N_SEEDS))
  MODEL="${ADDITIONAL_MODELS[$model_idx]}"
fi

EMBEDDING="${EMBEDDINGS[$embedding_idx]}"
CLASSIFICATION="${CLASSIFICATIONS[$classification_idx]}"
SEED="${SEEDS[$seed_idx]}"

echo "JOB_ID=$JOB_ID TASK_ID=$SGE_TASK_ID HOST=$(hostname) NSLOTS=${NSLOTS:-1}"
echo "Running: classification=$CLASSIFICATION model=$MODEL embedding=$EMBEDDING seed=$SEED"

# Permit the local debug script to test all task mappings without loading the
# cluster environment or fitting models.
if [[ "${CLASSIFICATION_MAP_ONLY:-0}" == "1" ]]; then
  echo "$CLASSIFICATION,$MODEL,$EMBEDDING,$SEED"
  exit 0
fi

mkdir -p output_files

. /etc/profile.d/modules.sh
module load anaconda
eval "$(conda shell.bash hook)"
conda activate /exports/eddie/scratch/s1948145/anaconda/envs/chem

echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
python --version

python -u 05_PY_SCRIPTS/run_classification_model_train.py \
  --model "$MODEL" \
  --classification "$CLASSIFICATION" \
  --embedding "$EMBEDDING" \
  --seed "$SEED" \
  --cv_splits 3 \
  --search_iterations 16 \
  --n_jobs "${NSLOTS:-1}" \
  --output_dir "$OUT_DIR"
