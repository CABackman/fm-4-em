#!/bin/bash
# SGE options
#$ -N gpu-chemprop_is_cv
#$ -cwd
#$ -q gpu
#$ -l gpu=1
#$ -l h_rt=24:00:00
#$ -m ae
#$ -o output_files/$JOB_NAME.$JOB_ID.out
#$ -e output_files/$JOB_NAME.$JOB_ID.err
#$ -M s1948145@ed.ac.uk

# Submit from the repository root:
#   qsub 03_SH_SCRIPTS/chemprop_train.sh
#
# This job runs nested CV for the IS dataset.  The outer test sets are the
# already-created files in 01_DATASETS/IS/random_seed*_split_test.npy.  The
# Python driver creates all inner folds and invokes Chemprop with explicit
# splits, so Chemprop's default random splitter is never used.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
OUT_DIR="${TRAIN_OUT_DIR:-/exports/eddie/scratch/s1948145/Model_ckpts/chemprop_is_cv}"

mkdir -p "${PROJECT_DIR}/output_files" "${OUT_DIR}"

. /etc/profile.d/modules.sh
module load anaconda
eval "$(conda shell.bash hook)"
conda activate chem

export OMP_NUM_THREADS="${NSLOTS:-1}"
export MKL_NUM_THREADS="${NSLOTS:-1}"
export OPENBLAS_NUM_THREADS="${NSLOTS:-1}"
export NUMEXPR_NUM_THREADS="${NSLOTS:-1}"

echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(command -v python)"
python --version
echo "Chemprop: $(command -v chemprop)"

python -u "${PROJECT_DIR}/05_PY_SCRIPTS/run_chemprop_is_cv.py" \
  --data-path "${PROJECT_DIR}/01_DATASETS/IS/IS.csv" \
  --split-dir "${PROJECT_DIR}/01_DATASETS/IS" \
  --output-dir "${OUT_DIR}" \
  --chemeleon-foundation "${PROJECT_DIR}/04_FM_CPS/chemeleon_mp.pt" \
  --outer-splits random_seed0 random_seed1 random_seed2 random_seed3 random_seed4 \
  --modes scratch chemeleon \
  --inner-folds "${CHEMPROP_INNER_FOLDS:-5}" \
  --hpo-trials "${CHEMPROP_HPO_TRIALS:-12}" \
  --hpo-epochs "${CHEMPROP_HPO_EPOCHS:-40}" \
  --hpo-patience "${CHEMPROP_HPO_PATIENCE:-12}" \
  --final-epochs "${CHEMPROP_FINAL_EPOCHS:-100}" \
  --final-patience "${CHEMPROP_FINAL_PATIENCE:-20}" \
  --batch-size "${CHEMPROP_BATCH_SIZE:-64}" \
  --num-workers "${CHEMPROP_NUM_WORKERS:-0}" \
  --accelerator "${CHEMPROP_ACCELERATOR:-auto}" \
  --devices "${CHEMPROP_DEVICES:-auto}" \
  --seed "${CHEMPROP_SEED:-2026}"
