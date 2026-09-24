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
**DOI**: 10.5281/zenodo.22939500


**Abstract**:
Abstract: This study evaluates the performance of off-the-shelf foundation model embeddings provided by CheMeleon, MolCLR, and MACE-OFF to predict energetic material properties from sparse datasets. Models were trained to predict crystal density, heat of formation and impact sensitivity from SMILES strings, with performance benchmarked against molecular representations provided by Morgan fingerprints and hand-crafted descriptors. Overall, the foundation model embeddings were found to be competitive with the hand-crafted descriptors, with MACE-OFF showing the strongest performance in numerical regression tasks. Heat of formation was found to be the most predictable property ( R 2 = 0.72 ), followed by crystal density ( R 2 = 0.54 ) and impact sensitivity ( R 2 = 0.47 ). Testing under more demanding cluster-based data splitting revealed that MACE-OFF retained the strongest performance across representations, suggesting that its learned embeddings generalise more effectively than hand-crafted descriptors to molecules that are more dissimilar to the training set distribution. Predictability for impact sensitivity improved noticeably when recast as a classification task, where CheMeleon emerged with comparable performance to the hand-crafted descriptors. These results show that while foundation model embeddings can provide useful low-input screening tools for energetic materials, they are not yet a competitive replacement for the bespoke models trained on larger datasets.


