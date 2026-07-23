#!/bin/bash
# SGE options
#$ -N maceQU_interpretability_tabfm
#$ -cwd
#$ -q gpu
#$ -l gpu=2
#$ -m ae
#$ -o output_files/$JOB_NAME.$JOB_ID.out
#$ -e output_files/$JOB_NAME.$JOB_ID.err
#$ -M s1948145@ed.ac.uk

set -euo pipefail

. /etc/profile.d/modules.sh
module load anaconda
conda activate /exports/eddie/scratch/s1948145/anaconda/envs/chem

echo "Running mace_explainable_tabpfm.py"
python 05_PY_SCRIPTS/mace_explainable_tabpfn.py
echo "Finished"

echo "Running QU_explainable_tabpfm.py"
python 05_PY_SCRIPTS/QU_explainable_tabpfn.py
echo "Finished"