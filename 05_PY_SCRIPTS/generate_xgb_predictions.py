#!/usr/bin/env python3
"""Regenerate test-set predictions for trained XGBoost models.

This script expects checkpoint artifacts laid out under the fixed directory
structure described in the project README. It traverses all XGBoost models
trained on the specified datasets/embeddings, reloads the saved estimators,
recomputes their predictions on the held-out test indices, and stores the
results in a single parquet file.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DATASETS = ["CD", "HOF", "IS"]
EMBEDDINGS = ("hq_descriptor", "maceoff_large")
OUTPUT_FILENAME = "deep_dive_xgb_predictions.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate pooled predictions from saved XGBoost checkpoints."
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help="Run directory containing <dataset>/xgboost/<embedding>/...",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(OUTPUT_FILENAME),
        help="Output Parquet path (default: %(default)s)",
    )
    return parser.parse_args()


def _load_dataset_table(dataset: str, datasets_dir: Path, cache: dict[str, pd.DataFrame], id_cache: dict[str, str]) -> tuple[pd.DataFrame, str]:
    if dataset not in cache:
        csv_path = datasets_dir / dataset / f"{dataset}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if "val" not in df.columns:
            raise KeyError(f"Missing 'val' column in {csv_path}")

        id_column_candidates = ("SMILES", "smiles", "ID", "id", "compound_id", "CompoundID")
        id_column = next((col for col in id_column_candidates if col in df.columns), None)
        if id_column is None:
            id_column = df.columns[0]

        if df[id_column].dtype.kind in {"O", "U", "S"}:
            df[id_column] = df[id_column].astype(str).str.strip()
        else:
            df[id_column] = df[id_column].astype(str)

        cache[dataset] = df
        id_cache[dataset] = id_column

    return cache[dataset], id_cache[dataset]


def _load_embeddings(dataset: str, embedding: str, embeddings_dir: Path, cache: dict[tuple[str, str], np.ndarray]) -> np.ndarray:
    key = (dataset, embedding)
    if key not in cache:
        emb_path = embeddings_dir / dataset / f"{embedding}_embeddings.npy"
        if not emb_path.exists():
            raise FileNotFoundError(f"Embedding array not found: {emb_path}")

        arr = np.load(emb_path)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        cache[key] = arr

    return cache[key]


def _load_test_indices(dataset: str, split_seed: int, datasets_dir: Path, cache: dict[tuple[str, int], np.ndarray]) -> np.ndarray:
    key = (dataset, split_seed)
    if key not in cache:
        idx_path = datasets_dir / dataset / f"random_seed{split_seed}_split_test.npy"
        if not idx_path.exists():
            raise FileNotFoundError(f"Test index file not found: {idx_path}")

        idx = np.load(idx_path)
        if idx.ndim != 1:
            idx = idx.ravel()
        cache[key] = idx.astype(int)

    return cache[key]


def _load_estimator(ckpt_path: Path):
    try:
        artifact = joblib.load(ckpt_path)
    except Exception:  # pragma: no cover - fallback to pickle if joblib fails
        with ckpt_path.open("rb") as fh:
            artifact = pickle.load(fh)

    if isinstance(artifact, dict):
        for key in ("best_estimator", "estimator", "model"):
            if key in artifact and hasattr(artifact[key], "predict"):
                return artifact[key]
        raise KeyError(f"No estimator with predict() found in checkpoint dict: {ckpt_path}")

    if not hasattr(artifact, "predict"):
        raise AttributeError(f"Loaded object has no predict(): {type(artifact)!r}")

    return artifact


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    datasets_dir = repo_root / "01_DATASETS"
    embeddings_dir = repo_root / "02_EMBEDDINGS"
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    data_cache: dict[str, pd.DataFrame] = {}
    id_cache: dict[str, str] = {}
    emb_cache: dict[tuple[str, str], np.ndarray] = {}
    test_idx_cache: dict[tuple[str, int], np.ndarray] = {}

    records: list[pd.DataFrame] = []
    checkpoints_found = False

    for dataset in DATASETS:
        for embedding in EMBEDDINGS:
            ckpt_dir = checkpoint_root / dataset / "xgboost" / embedding
            if not ckpt_dir.exists():
                raise FileNotFoundError(f"Checkpoint directory missing: {ckpt_dir}")

            for ckpt_path in sorted(ckpt_dir.glob("split=random_seed*_modelseed=*/ckpts/*.pkl")):
                checkpoints_found = True
                split_dir = ckpt_path.parent.parent.name
                match = re.match(r"split=random_seed(\d+)_modelseed=(\d+)", split_dir)
                if not match:
                    raise ValueError(f"Could not parse seeds from path: {ckpt_path}")
                split_seed = int(match.group(1))
                model_seed = int(match.group(2))

                estimator = _load_estimator(ckpt_path)

                df, id_column = _load_dataset_table(dataset, datasets_dir, data_cache, id_cache)
                embeddings = _load_embeddings(dataset, embedding, embeddings_dir, emb_cache)
                indices = _load_test_indices(dataset, split_seed, datasets_dir, test_idx_cache)

                if embeddings.shape[0] != len(df):
                    raise ValueError(
                        f"Row mismatch for {dataset}/{embedding}: "
                        f"embeddings have {embeddings.shape[0]} rows, dataset has {len(df)}"
                    )

                y_true = df["val"].to_numpy().astype(float)
                compound_ids = df.iloc[indices][id_column].to_list()
                X_test = embeddings[indices]
                y_test = y_true[indices]

                y_pred = estimator.predict(X_test)
                y_pred = np.asarray(y_pred).ravel()
                if y_pred.shape[0] != len(indices):
                    raise ValueError(
                        f"Prediction length mismatch at {ckpt_path}: "
                        f"got {y_pred.shape[0]}, expected {len(indices)}"
                    )

                record = pd.DataFrame(
                    {
                        "dataset": dataset,
                        "embedding": embedding,
                        "split_seed": split_seed,
                        "model_seed": model_seed,
                        "compound_id": compound_ids,
                        "y_true": y_test,
                        "y_pred": y_pred,
                    }
                )
                records.append(record)
                print(f"Processed {ckpt_path.relative_to(checkpoint_root)}")

    if not checkpoints_found:
        raise RuntimeError(
            f"No XGBoost checkpoints found under {checkpoint_root}."
        )

    all_results = pd.concat(records, ignore_index=True)
    all_results.to_parquet(output_path, index=False)
    print(f"Wrote {len(all_results)} predictions to {output_path}")


if __name__ == "__main__":
    main()
