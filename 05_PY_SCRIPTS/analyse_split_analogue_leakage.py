#!/usr/bin/env python
"""Quantify train-test analogue leakage for saved molecular data splits.

For every test molecule, this script calculates its maximum Tanimoto similarity
to any molecule in the corresponding training set.  This local, boundary-level
quantity is more directly relevant to analogue leakage than a global cluster
silhouette score.

The script expects the layout created by ``01_preprocess+splits.ipynb``::

    01_DATASETS/<dataset>/<dataset>.csv
    parser.add_argument("--datasets", nargs="+", default=["CD", "HOF", "IS"])
    01_DATASETS/<dataset>/cluster_label*_split_test.npy

Run from the repository root with the ``chem`` conda environment::

    conda run -n chem python 05_PY_SCRIPTS/analyse_split_analogue_leakage.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


METHODS = {
    "Random": "random_seed*_split_test.npy",
    "Cluster": "cluster_label*_split_test.npy",
}
COLORS = {"Random": "#0072B2", "Cluster": "#D55E00"}  # colour-blind safe
DISPLAY_NAMES = {"CD": r"$\rho_{cryst}$"}


def morgan_fingerprints(smiles: pd.Series, radius: int = 2, fp_size: int = 2048):
    """Return ECFP4 bit vectors, failing clearly on invalid SMILES."""
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=fp_size
    )
    fps = []
    for row, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(value))
        if mol is None:
            raise ValueError(f"Invalid SMILES at zero-based row {row}: {value!r}")
        fps.append(generator.GetFingerprint(mol))
    return fps


def nearest_train_similarities(fps, test_indices: np.ndarray) -> np.ndarray:
    """Maximum exact Tanimoto similarity from each test item to the train set."""
    n = len(fps)
    test_indices = np.asarray(test_indices, dtype=int)
    if test_indices.ndim != 1 or len(np.unique(test_indices)) != len(test_indices):
        raise ValueError("Test indices must be a one-dimensional array of unique rows")
    if len(test_indices) == 0 or test_indices.min() < 0 or test_indices.max() >= n:
        raise IndexError(f"Test indices are empty or outside [0, {n - 1}]")

    is_test = np.zeros(n, dtype=bool)
    is_test[test_indices] = True
    train_fps = [fp for i, fp in enumerate(fps) if not is_test[i]]
    if not train_fps:
        raise ValueError("A split has no training molecules")

    return np.asarray(
        [max(DataStructs.BulkTanimotoSimilarity(fps[i], train_fps)) for i in test_indices],
        dtype=float,
    )


def analyse_dataset(dataset_dir: Path, dataset_name: str) -> pd.DataFrame:
    """Calculate one row per test-molecule evaluation for both split methods."""
    frame = pd.read_csv(dataset_dir / dataset_name / f"{dataset_name}.csv")
    fps = morgan_fingerprints(frame["SMILES"])
    records = []

    for method, pattern in METHODS.items():
        files = sorted((dataset_dir / dataset_name).glob(pattern))
        if not files:
            raise FileNotFoundError(f"No {method} split files for {dataset_name}")
        for split_file in files:
            test_indices = np.load(split_file, allow_pickle=False)
            similarities = nearest_train_similarities(fps, test_indices)
            records.extend(
                {
                    "dataset": dataset_name,
                    "method": method,
                    "split": split_file.stem,
                    "test_index": int(index),
                    "max_train_similarity": float(similarity),
                    "nearest_train_distance": float(1.0 - similarity),
                }
                for index, similarity in zip(test_indices, similarities)
            )

    return pd.DataFrame.from_records(records)


def summarise_split_results(results: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    """Summarise at split level, the appropriate unit for method comparisons."""
    rows = []
    for keys, group in results.groupby(["dataset", "method", "split"], sort=False):
        row = {
            "dataset": keys[0],
            "method": keys[1],
            "split": keys[2],
            "n_test": len(group),
            "mean_max_similarity": group["max_train_similarity"].mean(),
            "median_max_similarity": group["max_train_similarity"].median(),
            "q25_max_similarity": group["max_train_similarity"].quantile(0.25),
            "q75_max_similarity": group["max_train_similarity"].quantile(0.75),
        }
        for threshold in thresholds:
            row[f"fraction_similarity_ge_{threshold:g}"] = (
                group["max_train_similarity"] >= threshold
            ).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def method_summary(split_summary: pd.DataFrame, leakage_threshold: float) -> pd.DataFrame:
    """Report median and range across the five saved folds/seeds."""
    leakage_col = f"fraction_similarity_ge_{leakage_threshold:g}"
    rows = []
    for keys, group in split_summary.groupby(["dataset", "method"], sort=False):
        rows.append(
            {
                "dataset": keys[0],
                "method": keys[1],
                "n_splits": len(group),
                "median_of_split_medians": group["median_max_similarity"].median(),
                "min_split_median": group["median_max_similarity"].min(),
                "max_split_median": group["median_max_similarity"].max(),
                f"median_{leakage_col}": group[leakage_col].median(),
                f"min_{leakage_col}": group[leakage_col].min(),
                f"max_{leakage_col}": group[leakage_col].max(),
            }
        )
    return pd.DataFrame(rows)


def plot_leakage_curves(
    results: pd.DataFrame,
    datasets: list[str],
    leakage_threshold: float,
    output_stem: Path,
) -> None:
    """Plot fold-balanced survival curves of nearest-train similarity."""
    thresholds = np.linspace(0.0, 1.0, 201)
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.45 * len(datasets), 3.25),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, dataset in zip(axes, datasets):
        subset = results[results["dataset"] == dataset]
        for method in METHODS:
            fold_curves = []
            method_data = subset[subset["method"] == method]
            for _, fold in method_data.groupby("split"):
                values = fold["max_train_similarity"].to_numpy()
                curve = np.asarray([(values >= t).mean() for t in thresholds])
                fold_curves.append(curve)
                ax.plot(thresholds, curve, color=COLORS[method], alpha=0.20, lw=0.8)

            # Every saved fold contributes equally, avoiding domination by a large cluster.
            mean_curve = np.mean(fold_curves, axis=0)
            ax.plot(thresholds, mean_curve, color=COLORS[method], lw=2.2)

        ax.axvline(leakage_threshold, color="0.35", lw=0.9, ls="--")
        ax.set_title(DISPLAY_NAMES.get(dataset, dataset), fontweight="normal")
        ax.set_xlabel(r"$T$")
        ax.set_xlim(0.2, 1.0)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="both", color="0.90", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel(r"Fraction of test molecules ≥ $T$")
    legend_handles = [
        Line2D([0], [0], color=COLORS[m], lw=2.2, label=m) for m in METHODS
    ] + [Line2D([0], [0], color="0.45", lw=0.8, alpha=0.45,
                label="Individual split")]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=3,
        columnspacing=1.6,
        handlelength=2.2,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["CD", "HOF", "IS"])
    parser.add_argument("--dataset-dir", type=Path, default=Path("01_DATASETS"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("07_OUTPUTS/split_leakage"))
    parser.add_argument("--leakage-threshold", type=float, default=0.70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = sorted(set([0.5, 0.6, args.leakage_threshold, 0.8, 0.9]))
    results = pd.concat(
        [analyse_dataset(args.dataset_dir, name) for name in args.datasets],
        ignore_index=True,
    )
    split_summary = summarise_split_results(results, thresholds)
    comparison = method_summary(split_summary, args.leakage_threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "nearest_train_similarity.csv", index=False)
    split_summary.to_csv(args.output_dir / "split_level_summary.csv", index=False)
    comparison.to_csv(args.output_dir / "method_comparison.csv", index=False)
    plot_leakage_curves(
        results,
        args.datasets,
        args.leakage_threshold,
        args.output_dir / "split_analogue_leakage",
    )

    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nOutputs written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
