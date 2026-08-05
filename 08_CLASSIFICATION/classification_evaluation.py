"""Reusable loading and aggregation helpers for classification evaluation."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CLASSIFICATIONS = ("binary", "tertiary", "quaternary", "quinary")
MODELS = ("svc", "logreg", "rf", "lightgbm", "bayes", "xgboost")
EMBEDDINGS = (
    "hq_descriptor",
    "chemeleon",
    "morgan_fps",
    "maceoff_large",
    "molclr",
)
SEEDS = (0, 1, 2, 3, 4)


def _metric(metrics: dict, name: str) -> float | None:
    value = metrics.get(name)
    return float(value) if value is not None else None


def read_summary(path: Path) -> dict:
    """Read one current-format summary into a flat evaluation record."""
    summary = json.loads(path.read_text(encoding="utf-8"))
    required = {"model", "embedding", "seed", "test_metrics"}
    missing = required.difference(summary)
    if missing:
        raise KeyError(f"{path} is missing fields: {sorted(missing)}")

    train = summary.get("train_metrics", {})
    test = summary["test_metrics"]
    classification = summary.get("classification", "binary")
    test_auc = test.get("roc_auc", test.get("roc_auc_ovr_macro"))
    train_auc = train.get("roc_auc", train.get("roc_auc_ovr_macro"))
    return {
        "classification": classification,
        "n_classes": len(summary.get("class_counts", {})) or None,
        "model": summary["model"],
        "embedding": summary["embedding"],
        "seed": int(summary["seed"]),
        "embedding_dim": summary.get("embedding_dim"),
        "test_size": summary.get("test_size"),
        "cv_splits": summary.get("cv_splits"),
        "n_train": summary.get("n_train"),
        "n_test": summary.get("n_test"),
        "best_cv_score": summary.get("best_cv_score"),
        "best_params": json.dumps(summary.get("best_params", {}), sort_keys=True),
        "train_accuracy": _metric(train, "accuracy"),
        "train_balanced_accuracy": _metric(train, "balanced_accuracy"),
        "train_f1_macro": _metric(train, "f1_macro"),
        "train_roc_auc": float(train_auc) if train_auc is not None else None,
        "test_accuracy": _metric(test, "accuracy"),
        "test_balanced_accuracy": _metric(test, "balanced_accuracy"),
        "test_precision_macro": _metric(test, "precision_macro"),
        "test_recall_macro": _metric(test, "recall_macro"),
        "test_f1_macro": _metric(test, "f1_macro"),
        "test_f1_weighted": _metric(test, "f1_weighted"),
        "test_roc_auc": float(test_auc) if test_auc is not None else None,
        "roc_auc_error": test.get("roc_auc_error"),
        "test_confusion_matrix": test.get("confusion_matrix"),
        "test_classification_report": test.get("classification_report", {}),
        "duration_seconds": summary.get("duration_seconds"),
        "started_at": summary.get("started_at"),
        "finished_at": summary.get("finished_at"),
        "run_dir": str(path.parent.resolve()),
        "summary_path": str(path.resolve()),
    }


def discover_runs(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discover summaries, returning latest unique runs and rejected files."""
    records: list[dict] = []
    rejected: list[dict] = []
    for path in sorted(results_root.glob("**/summary.json")):
        try:
            records.append(read_summary(path))
        except (OSError, ValueError, KeyError, TypeError) as error:
            rejected.append({"summary_path": str(path), "error": str(error)})
    if not records:
        raise FileNotFoundError(f"No readable summary.json files under {results_root}")

    all_runs = pd.DataFrame(records)
    all_runs["finished_at"] = pd.to_datetime(all_runs["finished_at"], errors="coerce")
    key = ["classification", "model", "embedding", "seed", "test_size"]
    runs = (
        all_runs.sort_values(["finished_at", "summary_path"], na_position="first")
        .drop_duplicates(key, keep="last")
        .sort_values(["classification", "model", "embedding", "seed"])
        .reset_index(drop=True)
    )
    return runs, pd.DataFrame(rejected, columns=["summary_path", "error"])


def completion_table(
    runs: pd.DataFrame,
    classifications: Iterable[str] = CLASSIFICATIONS,
    models: Iterable[str] = MODELS,
    embeddings: Iterable[str] = EMBEDDINGS,
    seeds: Iterable[int] = SEEDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return every expected task with status plus the incomplete subset."""
    columns = ["classification", "model", "embedding", "seed"]
    expected = pd.DataFrame(
        product(classifications, models, embeddings, seeds), columns=columns
    )
    observed = runs[columns].drop_duplicates().assign(completed=True)
    completion = expected.merge(observed, on=columns, how="left")
    completion["completed"] = completion["completed"].eq(True)
    missing = completion.loc[~completion["completed"], columns].reset_index(drop=True)
    return completion, missing


def aggregate_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate held-out metrics across available seeds without imputing runs."""
    grouped = (
        runs.assign(
            balanced_accuracy_gap=(
                runs["train_balanced_accuracy"] - runs["test_balanced_accuracy"]
            )
        )
        .groupby(["classification", "model", "embedding"], as_index=False)
        .agg(
            seeds_completed=("seed", "nunique"),
            balanced_accuracy_mean=("test_balanced_accuracy", "mean"),
            balanced_accuracy_std=("test_balanced_accuracy", "std"),
            accuracy_mean=("test_accuracy", "mean"),
            accuracy_std=("test_accuracy", "std"),
            f1_macro_mean=("test_f1_macro", "mean"),
            f1_macro_std=("test_f1_macro", "std"),
            roc_auc_mean=("test_roc_auc", "mean"),
            roc_auc_std=("test_roc_auc", "std"),
            generalisation_gap_mean=("balanced_accuracy_gap", "mean"),
            duration_seconds_mean=("duration_seconds", "mean"),
        )
    )
    return grouped.sort_values(
        ["classification", "balanced_accuracy_mean"],
        ascending=[True, False],
    ).reset_index(drop=True)


def normalised_confusion_matrix(group: pd.DataFrame, n_classes: int) -> np.ndarray:
    """Sum compatible confusion matrices and normalise each true-class row."""
    matrices = []
    for value in group["test_confusion_matrix"]:
        matrix = np.asarray(value, dtype=float)
        if matrix.shape == (n_classes, n_classes):
            matrices.append(matrix)
    if not matrices:
        return np.full((n_classes, n_classes), np.nan)
    total = np.sum(matrices, axis=0)
    row_totals = total.sum(axis=1, keepdims=True)
    return np.divide(total, row_totals, out=np.zeros_like(total), where=row_totals != 0)


def per_class_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    """Extract class-level held-out metrics from sklearn reports."""
    rows = []
    for run in runs.itertuples(index=False):
        for label, metrics in run.test_classification_report.items():
            if label in {"accuracy", "macro avg", "weighted avg"}:
                continue
            if not isinstance(metrics, dict):
                continue
            rows.append(
                {
                    "classification": run.classification,
                    "model": run.model,
                    "embedding": run.embedding,
                    "seed": run.seed,
                    "class_label": label,
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1-score"),
                    "support": metrics.get("support"),
                }
            )
    return pd.DataFrame(rows)


def load_predictions(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load prediction CSVs and compute per-molecule error summaries."""
    frames = []
    for run in runs.itertuples(index=False):
        path = Path(run.run_dir) / "test_predictions.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        target_candidates = (run.classification, f"{run.classification}_class")
        target = next((name for name in target_candidates if name in frame.columns), None)
        if target is None or "predicted_class" not in frame.columns:
            continue
        frame["classification"] = run.classification
        frame["model"] = run.model
        frame["embedding"] = run.embedding
        frame["seed"] = run.seed
        frame["true_class"] = frame[target]
        frame["incorrect"] = frame["predicted_class"] != frame["true_class"]
        frames.append(frame)

    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    predictions = pd.concat(frames, ignore_index=True)
    identity = ["classification", "row_index", "true_class"]
    difficult = (
        predictions.groupby(identity, as_index=False)
        .agg(
            evaluations=("incorrect", "size"),
            errors=("incorrect", "sum"),
            error_rate=("incorrect", "mean"),
        )
        .sort_values(["classification", "error_rate", "evaluations"],
                     ascending=[True, False, False])
        .reset_index(drop=True)
    )
    metadata_columns = [
        column
        for column in ("row_index", "SMILES", "canonical_SMILES", "Source", "Exp IS (J)")
        if column in predictions.columns
    ]
    if metadata_columns:
        metadata = predictions[metadata_columns].drop_duplicates("row_index")
        difficult = difficult.merge(metadata, on="row_index", how="left")
    return predictions, difficult


def export_tables(
    export_dir: Path,
    runs: pd.DataFrame,
    aggregate: pd.DataFrame,
    missing: pd.DataFrame,
    rejected: pd.DataFrame,
    predictions: pd.DataFrame,
    difficult: pd.DataFrame,
    class_metrics: pd.DataFrame,
) -> None:
    """Write evaluation tables without serialising nested confusion matrices."""
    export_dir.mkdir(parents=True, exist_ok=True)
    runs.drop(columns=["test_confusion_matrix", "test_classification_report"]).to_csv(
        export_dir / "classification_runs.csv", index=False
    )
    aggregate.to_csv(export_dir / "classification_aggregate.csv", index=False)
    missing.to_csv(export_dir / "classification_missing_runs.csv", index=False)
    rejected.to_csv(export_dir / "classification_rejected_summaries.csv", index=False)
    class_metrics.to_csv(export_dir / "classification_per_class_metrics.csv", index=False)
    if not predictions.empty:
        predictions.to_csv(export_dir / "classification_test_predictions.csv", index=False)
    if not difficult.empty:
        difficult.to_csv(export_dir / "classification_difficult_molecules.csv", index=False)
