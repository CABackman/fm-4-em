# Foundation models for energetic materials

Anonymous code accompanying a NeurIPS submission. This repository compares
frozen molecular representations for three energetic-material regression tasks:

- crystal density (CD);
- solid-state heat of formation (HOF);
- impact sensitivity (IS).

The representations are CheMeleon, MolCLR, MACE-OFF, Morgan fingerprints, and
17 hand-crafted molecular descriptors. They are evaluated with linear
regression, random forest, XGBoost, and multilayer perceptron models.

## Setup

The development environment uses Python 3.11:

```bash
conda env create -f environment.yml
conda activate chem
```

Precomputed datasets, embeddings, and split indices are included, so the
foundation-model embeddings do not need to be regenerated.

## Run one experiment

From the repository root:

```bash
python 05_PY_SCRIPTS/run_model_train.py \
  --dataset HOF \
  --embedding hq_descriptor \
  --split cluster_label0 \
  --model linear \
  --seed 0 \
  --output_dir 07_OUTPUTS
```

Available values are:

- datasets: `CD`, `HOF`, `IS`;
- embeddings: `chemeleon`, `molclr`, `maceoff_large`, `morgan_fps`,
  `hq_descriptor`;
- models: `linear`, `rf`, `xgboost`, `mlp`;
- splits: `random_seed0` through `random_seed4`, and `cluster_label0`
  through `cluster_label4`;
- model seeds: 0 through 4.

Results are saved under `07_OUTPUTS/YYYYMMDD/`. The full grid can be submitted
through the SGE example in `03_SH_SCRIPTS/train_all.sh` or by looping over the
values above.

## Repository structure

- `01_DATASETS/`: processed datasets and test indices
- `02_EMBEDDINGS/`: precomputed molecular representations
- `03_SH_SCRIPTS/`: training launchers
- `05_PY_SCRIPTS/`: training and results scripts
- `06_NOTEBOOKS/`: preprocessing, embedding, visualization, and analysis
- `modules/`: model and data utilities

The experiments use five random 80/20 splits and five structure-cluster
holdouts. Hyperparameter selection is performed only on the training data. The
main reported metrics are MAE, RMSE, and R-squared.

## Submission note

Notebook outputs and machine-specific paths have been removed. Do not include
`.git/`, `07_OUTPUTS/`, caches, or local model directories in the anonymous
submission. After committing the intended files, create the ZIP with:

```bash
git archive --format=zip --output ../anonymous_neurips_code.zip HEAD
```

The datasets originate from published literature compilations. Their citations
and redistribution terms should be included in the paper and finalized before
any public release.
