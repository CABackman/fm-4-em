# Frozen molecular representations for energetic-material property prediction

Anonymous code and data supplement for a NeurIPS submission. This repository
benchmarks frozen molecular representations in low-data regression of crystal
density (CD), solid-state heat of formation (HOF), and impact sensitivity (IS).

The review artifact intentionally contains no author names, affiliations,
acknowledgements, submission number, DOI for the submitted paper, or identifying
repository URL. Add those only after the review process.

## Method at a glance

Five representations are paired with four regression heads:

- representations: CheMeleon, MolCLR, MACE-OFF (large), 17 hand-crafted
  molecular descriptors, and Morgan/ECFP4 fingerprints;
- heads: elastic-net linear regression, random forest, XGBoost, and a
  multilayer perceptron (MLP);
- evaluation: five 80/20 random splits and five structure-cluster holdouts;
- model seeds: 0, 1, 2, 3, and 4;
- selection metric: mean absolute error (MAE), with hyperparameter selection
  performed only within the training partition.

The structure-cluster tests hold out one of five clusters produced from Morgan
fingerprints after dimensionality reduction. Their test sizes vary with cluster
membership and are not intended to be 20% exactly.

## Artifact contents

| Path | Contents |
| --- | --- |
| `01_DATASETS/` | Processed targets and the exact test indices used in the paper |
| `02_EMBEDDINGS/` | Precomputed representations aligned row-for-row with each processed CSV |
| `03_SH_SCRIPTS/` | Local smoke test and an optional SGE array launcher |
| `05_PY_SCRIPTS/run_model_train.py` | Main training and evaluation entry point |
| `05_PY_SCRIPTS/generate_xgb_predictions.py` | Reconstruct pooled predictions from saved XGBoost checkpoints |
| `06_NOTEBOOKS/` | Preprocessing, representation generation, visualization, and result analysis |
| `modules/` | Split, representation, model, metric, and plotting implementations |
| `environment.yml` | Exact Python 3.11 Conda environment used for development |

`07_OUTPUTS/`, caches, editor settings, external model repositories, and local
foundation-model checkpoints are ignored and must not be placed in the review
archive. The supplied processed data, split files, and embeddings are sufficient
to rerun the downstream benchmark without regenerating representations.

## Data

| Task | Rows | Target stored in `val` | Random test rows | Cluster test rows |
| --- | ---: | --- | --- | --- |
| CD | 456 | crystal density in g cm^-3 | 92 per split | 112, 47, 83, 122, 92 |
| HOF | 205 | experimental solid-state heat of formation in kJ mol^-1 | 41 per split | 22, 57, 44, 49, 33 |
| IS | 933 | natural logarithm of impact energy in joules | 187 per split | 451, 149, 166, 81, 86 |

The CSV row order is part of the artifact contract: every embedding matrix and
split-index array indexes that order. Do not sort or otherwise reorder a CSV.
All three tasks contain molecular records, not human-subject data. The raw and
intermediate tables retained in `01_DATASETS/` document the merges used to
construct the processed tables. `06_NOTEBOOKS/01_preprocess+splits.ipynb`
contains the preprocessing and split-generation logic.

The impact-sensitivity data combine published compilations, including the
Mathieu data (DOI `10.1016/j.jmgm.2015.09.001`) and the dataset associated with
DOI `10.1039/D5DD00357A`; record-level literature references are retained in
the raw CSV files. These names and citations identify third-party scholarly
sources, not the anonymous submitters.

## Representations

| Representation name used by CLI | CD shape | HOF shape | IS shape |
| --- | --- | --- | --- |
| `chemeleon` | 456 x 2048 | 205 x 2048 | 933 x 2048 |
| `molclr` | 456 x 512 | 205 x 512 | 933 x 512 |
| `maceoff_large` | 456 x 448 | 205 x 448 | 933 x 448 |
| `hq_descriptor` | 456 x 17 | 205 x 17 | 933 x 17 |
| `morgan_fps` | 456 x 2048 | 205 x 2048 | 933 x 2048 |

The precomputed arrays are the canonical inputs for reproducing the reported
downstream results. Regenerating CheMeleon, MolCLR, or MACE-OFF embeddings can
vary with upstream checkpoints, conformer generation, package versions, and
hardware. The exploratory regeneration code is in
`06_NOTEBOOKS/02_get_embeddings.ipynb`, but the external MolCLR repository and
checkpoint are not included in this review archive.

## Installation

Linux and Conda are assumed. From the repository root:

```bash
conda env create -f environment.yml
conda activate chem
```

The locked environment uses Python 3.11 and includes the CPU/GPU scientific
stack used by all stages. Creating it requires substantial disk space because
it includes PyTorch and CUDA packages. For downstream training only, a smaller
environment can be constructed with Python 3.11, NumPy, pandas, scikit-learn,
XGBoost, joblib, PyTorch, and PyYAML; the exact versions remain recorded in
`environment.yml`.

## Quick reproducibility check

This CPU-friendly command runs one paper configuration with the supplied HOF
data, cluster split, and descriptor matrix:

```bash
python 05_PY_SCRIPTS/run_model_train.py \
  --dataset HOF \
  --embedding hq_descriptor \
  --split cluster_label0 \
  --model linear \
  --seed 0 \
  --output_dir 07_OUTPUTS
```

Outputs are written beneath:

```text
07_OUTPUTS/YYYYMMDD/HOF/linear/hq_descriptor/
  split=cluster_label0_modelseed=0/
    ckpts/
    summary.json
    summary_seed=0.json
```

Each summary records the dataset, embedding dimension, split, model seed,
metrics, timestamp, and input paths. Generated paths are local provenance and
must be audited before sharing newly generated output files.

## Reproduce the full downstream benchmark

The following loop runs all 3 tasks x 5 representations x 4 heads x 10 splits
x 5 model seeds (3,000 fits). Run it from the repository root. This is a large
CPU workload; parallelize at the job level on available infrastructure.

```bash
for dataset in CD HOF IS; do
  for embedding in hq_descriptor chemeleon molclr maceoff_large morgan_fps; do
    for model in linear rf xgboost mlp; do
      for split in \
        random_seed0 random_seed1 random_seed2 random_seed3 random_seed4 \
        cluster_label0 cluster_label1 cluster_label2 cluster_label3 cluster_label4; do
        for seed in 0 1 2 3 4; do
          python 05_PY_SCRIPTS/run_model_train.py \
            --dataset "$dataset" \
            --embedding "$embedding" \
            --model "$model" \
            --split "$split" \
            --seed "$seed" \
            --output_dir 07_OUTPUTS
        done
      done
    done
  done
done
```

`03_SH_SCRIPTS/train_all.sh` provides an SGE array example for cluster splits.
Set `CONDA_ENV` and `OUTPUT_DIR` for the target system; no institution-specific
host, account, or filesystem path is assumed.

To regenerate the pooled XGBoost prediction table from an existing dated run:

```bash
python 05_PY_SCRIPTS/generate_xgb_predictions.py \
  --checkpoint-root 07_OUTPUTS/YYYYMMDD \
  --output 07_OUTPUTS/deep_dive_xgb_predictions.parquet
```

Use `06_NOTEBOOKS/04_results_processing.ipynb` for result aggregation and paper
figures. Notebook outputs and execution metadata are deliberately cleared to
prevent identity leakage; rerunning the notebook will populate them locally.

## Experimental details

- Random test partitions contain 20% of each dataset and use split seeds 0--4.
- Cluster partitions use five Morgan-fingerprint-derived clusters and hold out
  one complete cluster at a time.
- Linear models use five-fold training-only cross-validation over elastic-net
  regularization parameters.
- Random forest and XGBoost use three-fold randomized search with 20 trials.
- MLP tuning uses three training-only folds, 20 sampled configurations, early
  stopping, and a final refit; its full search space and budgets are defined in
  `modules/mlp_tuning.py`.
- The held-out test partition is evaluated only after model selection.
- Primary metrics written by the models are MAE, RMSE, and R-squared.
- Seeds control split selection, cross-validation shuffling, model fitting, and
  MLP initialization. Exact bitwise equality across package versions or GPU
  types is not guaranteed.

No reliable end-to-end wall-clock or energy measurement was retained. Runtime
depends strongly on representation dimension, dataset, model family, and
hardware; the linear quick check is the recommended artifact sanity test.

## Limitations and responsible use

The datasets are small, heterogeneous literature compilations. Experimental
protocols and measurement noise differ across sources, especially for impact
sensitivity. Random splits can overstate generalization when close structural
analogues occur in both partitions; cluster splits are included to expose this
effect. The models predict molecular-level tabular targets and do not model
crystal defects, formulation, processing history, or experimental conditions.

Energetic-material property prediction is dual use. This artifact is intended
for scientific benchmarking and hazard-aware materials research. Predictions
must not substitute for regulated safety assessment, expert review, or physical
testing, and the repository provides no synthesis or operational instructions.

## Licenses and third-party assets

No top-level software license was present when this anonymous review artifact
was prepared. Consequently, this package is supplied for confidential peer
review, not as a grant of public redistribution rights. Before a public or
camera-ready release, the authors must add an appropriate code license and
verify redistribution terms for every included dataset, pretrained checkpoint,
and derived embedding. In particular, the dataset associated with DOI
`10.1039/D5DD00357A` is published under CC BY-NC 3.0, while the terms of the
other compiled records must be checked at their cited sources.

Upstream projects and papers retain their own licenses and citation
requirements. The archive includes derived arrays and one CheMeleon checkpoint,
but it does not vendor the MolCLR or MACE-OFF source repositories. This section
is a disclosure, not a replacement for those upstream terms.
