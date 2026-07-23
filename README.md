# Foundation models for energetic materials

This repository benchmarks frozen molecular foundation-model embeddings for low-data prediction of energetic-material properties. The tasks are crystal density, solid-state heat of formation, and impact sensitivity for molecular energetic materials.

## Benchmark

The molecular representations are:

- CheMeleon, MolCLR, and MACE-OFF embeddings
- Morgan (ECFP4) fingerprints
- Quayle hand-crafted energetic-material descriptors

Each representation is evaluated with linear regression, random forest, XGBoost, and multilayer perceptron prediction heads.

Models are tested on five 80/20 random train/test splits and five structure-based splits that hold out clusters defined from Morgan fingerprints. The cluster splits reduce leakage between close molecular analogues and test performance on structurally distinct molecules.

## Key result

MACE-OFF performs best overall, while the Quayle descriptors are the most robust under structure-based splits. All representations perform worse on the harder cluster splits, and numerical impact-sensitivity prediction remains limited by noisy data.

## Repository contents

- `01_DATASETS/`: curated datasets and saved split indices
- `02_EMBEDDINGS/`: precomputed molecular representations
- `05_PY_SCRIPTS/`: training and analysis scripts
- `06_NOTEBOOKS/`: preprocessing, embedding, and results notebooks
- `07_OUTPUTS/`: model outputs and evaluation results (not tracked on git)

## Paper Abstract & Link
**DOI**:
**Abstract**:


