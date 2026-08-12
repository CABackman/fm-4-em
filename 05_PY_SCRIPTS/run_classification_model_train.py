"""Train classification models on the embeddings in ``08_CLASSIFICATION``.

The script is designed for local use and independent SGE array tasks. Each run
handles one classification, model, embedding, and random seed. It loads a fixed
held-out split, tunes hyperparameters only on the training portion, evaluates the
selected model once on the held-out test portion, and saves all artifacts.

Example
-------
python 05_PY_SCRIPTS/run_classification_model_train.py \
    --model logreg \
    --embedding molclr \
    --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


REPO_ROOT = Path(__file__).resolve().parents[1]


MODEL_NAMES = ("svc", "logreg", "rf", "lightgbm", "bayes", "xgboost")
CLASSIFICATION_NAMES = ("binary", "tertiary", "quaternary", "quinary")


def make_pipeline(classifier: Any) -> Pipeline:
    """Build the same scaler/classifier structure used by the notebook models."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one classifier on one classification embedding."
    )
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument(
        "--classification",
        choices=CLASSIFICATION_NAMES,
        default="binary",
        help="Classification target and fixed-split directory to use.",
    )
    parser.add_argument(
        "--embedding",
        required=True,
        help=(
            "Embedding name, such as molclr. The default inferred filename is "
            "08_CLASSIFICATION/02_EMBEDDINGS/<name>_embeddings.npy."
        ),
    )
    parser.add_argument(
        "--classification_dir",
        default="08_CLASSIFICATION",
        help="Classification data directory relative to the project root.",
    )
    parser.add_argument(
        "--data_csv",
        default=None,
        help="Optional explicit dataset CSV path.",
    )
    parser.add_argument(
        "--embedding_path",
        default=None,
        help="Optional explicit embedding .npy path.",
    )
    parser.add_argument(
        "--split_path",
        default=None,
        help="Optional explicit .npy file containing held-out row positions.",
    )
    parser.add_argument(
        "--target_column",
        default=None,
        help=(
            "Target column override. By default, use --classification, falling "
            "back to '<classification>_class' for compatible older datasets."
        ),
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.20,
        help="Fraction reserved for final stratified testing.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cv_splits",
        type=int,
        default=3,
        help="Training-only stratified CV folds used for model selection.",
    )
    parser.add_argument(
        "--search_iterations",
        type=int,
        default=16,
        help="Maximum randomized-search candidates for tree models.",
    )
    parser.add_argument(
        "--scoring",
        default="balanced_accuracy",
        help="Scikit-learn scorer used during training-only model selection.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=None,
        help="Parallel CV workers. Defaults to NSLOTS, or 1 outside SGE.",
    )
    parser.add_argument(
        "--output_dir",
        default="07_OUTPUTS/classification",
        help="Output root, either absolute or relative to the project root.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Validate inputs and the fixed split, then exit without fitting.",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Use one hyperparameter candidate for a minimal end-to-end test.",
    )
    return parser.parse_args()


def resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def infer_embedding_path(classification_dir: Path, embedding: str) -> Path:
    if Path(embedding).name != embedding:
        raise ValueError(
            "--embedding must be a name, not a path. Use --embedding_path "
            "for an explicit path."
        )

    filename = embedding if embedding.endswith(".npy") else f"{embedding}_embeddings.npy"
    return classification_dir / "02_EMBEDDINGS" / filename


def resolve_target_column(
    data: pd.DataFrame,
    requested: str | None,
    classification: str,
) -> str:
    """Resolve current and legacy target-column naming conventions."""
    if requested is not None:
        candidates = [requested]
    else:
        candidates = [classification, f"{classification}_class"]

    for candidate in candidates:
        if candidate in data.columns:
            return candidate
    raise KeyError(
        f"None of the target columns {candidates!r} are present. "
        f"Available columns: {data.columns.tolist()}"
    )


def load_inputs(
    dataset_path: Path,
    embedding_path: Path,
    target_column: str | None,
    classification: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, str]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Classification dataset not found: {dataset_path}")
    if not embedding_path.is_file():
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

    data = pd.read_csv(dataset_path)
    target_column = resolve_target_column(data, target_column, classification)
    if data[target_column].isna().any():
        raise ValueError(f"Target column {target_column!r} contains missing values.")

    y = data[target_column].to_numpy()
    embeddings = np.load(embedding_path)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, 1)
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding matrix; got shape {embeddings.shape}."
        )
    if embeddings.shape[0] != len(data):
        raise ValueError(
            f"Embedding rows ({embeddings.shape[0]}) do not match dataset "
            f"rows ({len(data)})."
        )
    if not np.issubdtype(embeddings.dtype, np.number):
        raise TypeError(f"Embeddings must be numeric; got {embeddings.dtype}.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain missing or infinite values.")
    if np.unique(y).size < 2:
        raise ValueError("Classification requires at least two target classes.")

    return data, embeddings, y, target_column


def load_test_indices(split_path: Path, n_samples: int) -> np.ndarray:
    """Load and validate positional row indices for the fixed test split."""
    if not split_path.is_file():
        raise FileNotFoundError(f"Held-out split not found: {split_path}")
    test_idx = np.asarray(np.load(split_path))
    if test_idx.ndim != 1 or not np.issubdtype(test_idx.dtype, np.integer):
        raise ValueError("Held-out indices must be a one-dimensional integer array.")
    test_idx = test_idx.astype(int, copy=False)
    if test_idx.size == 0:
        raise ValueError("Held-out index array is empty.")
    if np.unique(test_idx).size != test_idx.size:
        raise ValueError("Held-out index array contains duplicates.")
    if test_idx.min() < 0 or test_idx.max() >= n_samples:
        raise IndexError(
            f"Held-out indices must be in [0, {n_samples - 1}], got "
            f"[{test_idx.min()}, {test_idx.max()}]."
        )
    return test_idx


def make_search(
    model_name: str,
    seed: int,
    cv: StratifiedKFold,
    scoring: str,
    n_jobs: int,
    search_iterations: int,
    n_train_samples: int,
    n_features: int,
    smoke_test: bool = False,
):
    common_search_args = {
        "cv": cv,
        "scoring": scoring,
        "n_jobs": n_jobs,
        "refit": True,
        "return_train_score": False,
        "error_score": "raise",
    }

    if model_name == "svc":
        # LinearSVC's dual parameter must be a boolean in older sklearn
        # versions. Use the dual formulation only when a typical inner-CV
        # training fold has fewer rows than feature columns.
        inner_train_size = n_train_samples * (cv.n_splits - 1) / cv.n_splits
        use_dual = n_features > inner_train_size
        estimator = make_pipeline(
            LinearSVC(
                random_state=seed,
                max_iter=100_000,
                dual=use_dual,
            )
        )
        param_grid = {
            "classifier__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "classifier__class_weight": [None, "balanced"],
        }
        if smoke_test:
            param_grid = {"classifier__C": [1.0], "classifier__class_weight": [None]}
        return GridSearchCV(estimator, param_grid, **common_search_args)

    if model_name == "logreg":
        estimator = make_pipeline(
            LogisticRegression(
                random_state=seed,
                max_iter=10_000,
                solver="lbfgs",
            )
        )
        param_grid = {
            "classifier__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "classifier__class_weight": [None, "balanced"],
        }
        if smoke_test:
            param_grid = {"classifier__C": [1.0], "classifier__class_weight": [None]}
        return GridSearchCV(estimator, param_grid, **common_search_args)

    if model_name == "bayes":
        estimator = make_pipeline(GaussianNB())
        param_grid = {
            "classifier__var_smoothing": np.logspace(-12, -6, 13).tolist(),
        }
        if smoke_test:
            param_grid = {"classifier__var_smoothing": [1e-9]}
        return GridSearchCV(estimator, param_grid, **common_search_args)

    if model_name == "rf":
        estimator = make_pipeline(
            RandomForestClassifier(
                random_state=seed,
                n_jobs=1,
            )
        )
        param_distributions = {
            "classifier__n_estimators": [200, 400, 600],
            "classifier__max_depth": [None, 8, 16, 32],
            "classifier__max_features": ["sqrt", "log2", 0.5],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__class_weight": [None, "balanced"],
        }
        if smoke_test:
            param_distributions = {
                "classifier__n_estimators": [5],
                "classifier__max_depth": [3],
                "classifier__max_features": ["sqrt"],
                "classifier__min_samples_leaf": [1],
                "classifier__class_weight": [None],
            }
        return RandomizedSearchCV(
            estimator,
            param_distributions,
            n_iter=search_iterations,
            random_state=seed,
            **common_search_args,
        )

    if model_name == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as error:
            raise ImportError(
                "The lightgbm model was requested, but LightGBM is not installed "
                "in the active environment. Install lightgbm or remove it from "
                "the HPC MODELS array."
            ) from error

        estimator = make_pipeline(
            lgb.LGBMClassifier(
                random_state=seed,
                n_jobs=1,
                verbosity=-1,
            )
        )
        param_distributions = {
            "classifier__n_estimators": [200, 400, 600],
            "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "classifier__num_leaves": [15, 31, 63, 127],
            "classifier__max_depth": [-1, 6, 10, 16],
            "classifier__colsample_bytree": [0.6, 0.8, 1.0],
            "classifier__subsample": [0.7, 0.85, 1.0],
            "classifier__class_weight": [None, "balanced"],
        }
        if smoke_test:
            param_distributions = {
                "classifier__n_estimators": [5],
                "classifier__learning_rate": [0.1],
                "classifier__num_leaves": [15],
                "classifier__max_depth": [3],
                "classifier__colsample_bytree": [1.0],
                "classifier__subsample": [1.0],
                "classifier__class_weight": [None],
            }
        return RandomizedSearchCV(
            estimator,
            param_distributions,
            n_iter=search_iterations,
            random_state=seed,
            **common_search_args,
        )

    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError(
                "The xgboost model was requested, but XGBoost is not installed "
                "in the active environment. Install xgboost or remove it from "
                "the HPC model list."
            ) from error

        estimator = make_pipeline(
            XGBClassifier(
                random_state=seed,
                n_jobs=1,
                eval_metric="logloss",
            )
        )
        param_distributions = {
            "classifier__n_estimators": [100, 200, 400, 600],
            "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "classifier__max_depth": [3, 5, 7, 10],
            "classifier__min_child_weight": [1, 3, 5],
            "classifier__subsample": [0.7, 0.85, 1.0],
            "classifier__colsample_bytree": [0.6, 0.8, 1.0],
        }
        if smoke_test:
            param_distributions = {
                "classifier__n_estimators": [5],
                "classifier__learning_rate": [0.1],
                "classifier__max_depth": [3],
                "classifier__min_child_weight": [1],
                "classifier__subsample": [1.0],
                "classifier__colsample_bytree": [1.0],
            }
        return RandomizedSearchCV(
            estimator,
            param_distributions,
            n_iter=search_iterations,
            random_state=seed,
            **common_search_args,
        )

    raise ValueError(f"Unknown model {model_name!r}; choose from {MODEL_NAMES}.")


def _prediction_scores(model, x: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return None


def calculate_metrics(model, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    predictions = model.predict(x)
    labels = np.asarray(model.classes_)
    metrics: dict[str, Any] = {
        "n_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "precision_macro": float(
            precision_score(y, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y, predictions, average="macro")),
        "f1_weighted": float(f1_score(y, predictions, average="weighted")),
        "confusion_matrix": confusion_matrix(y, predictions, labels=labels).tolist(),
        "classification_report": classification_report(
            y,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }

    scores = _prediction_scores(model, x)
    if scores is not None:
        try:
            if len(labels) == 2:
                positive_scores = scores[:, 1] if scores.ndim == 2 else scores
                metrics["roc_auc"] = float(roc_auc_score(y, positive_scores))
            else:
                metrics["roc_auc_ovr_macro"] = float(
                    roc_auc_score(
                        y,
                        scores,
                        labels=labels,
                        average="macro",
                        multi_class="ovr",
                    )
                )
        except ValueError as error:
            metrics["roc_auc_error"] = str(error)

    return metrics


def json_ready(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialise {type(value).__name__}")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test_size must lie strictly between 0 and 1.")
    if args.cv_splits < 2:
        raise ValueError("--cv_splits must be at least 2.")
    if args.search_iterations < 1:
        raise ValueError("--search_iterations must be at least 1.")

    n_jobs = args.n_jobs
    if n_jobs is None:
        n_jobs = int(os.environ.get("NSLOTS", "1"))
    if n_jobs < 1:
        raise ValueError("--n_jobs must be at least 1.")

    base_dir = Path(os.environ.get("PROJECT_DIR", REPO_ROOT)).resolve()
    classification_dir = resolve_path(base_dir, args.classification_dir)
    dataset_path = (
        resolve_path(base_dir, args.data_csv)
        if args.data_csv
        else classification_dir / "01_DATASETS" / "combined_IS_dataset.csv"
    )
    embedding_path = (
        resolve_path(base_dir, args.embedding_path)
        if args.embedding_path
        else infer_embedding_path(classification_dir, args.embedding)
    )
    split_path = (
        resolve_path(base_dir, args.split_path)
        if args.split_path
        else classification_dir
        / "01_DATASETS"
        / args.classification
        / f"random_seed{args.seed}_split_test.npy"
    )
    output_root = resolve_path(base_dir, args.output_dir)

    print(f"Project root:      {base_dir}")
    print(f"Dataset CSV:       {dataset_path}")
    print(f"Embedding file:    {embedding_path}")
    print(f"Held-out split:    {split_path}")
    print(f"Classification:    {args.classification}")
    print(f"Model:             {args.model}")
    print(f"Seed:              {args.seed}")
    print(f"CV workers:        {n_jobs}")

    data, embeddings, y, target_column = load_inputs(
        dataset_path,
        embedding_path,
        args.target_column,
        args.classification,
    )
    print(f"Target column:     {target_column}")
    indices = np.arange(len(y), dtype=int)
    test_idx = load_test_indices(split_path, len(y))
    train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)

    expected_n_test = int(np.ceil(args.test_size * len(y)))
    if len(test_idx) != expected_n_test:
        raise ValueError(
            f"Fixed split has {len(test_idx)} rows, but --test_size={args.test_size:g} "
            f"implies {expected_n_test} rows for {len(y)} samples."
        )
    full_classes = np.unique(y)
    if not np.array_equal(np.unique(y[train_idx]), full_classes):
        raise ValueError("The training split does not contain every target class.")
    if not np.array_equal(np.unique(y[test_idx]), full_classes):
        raise ValueError("The test split does not contain every target class.")

    if args.validate_only:
        print(
            f"Validation passed: {len(train_idx)} train rows, {len(test_idx)} test "
            f"rows, {len(full_classes)} classes."
        )
        return

    train_class_counts = pd.Series(y[train_idx]).value_counts()
    if int(train_class_counts.min()) < args.cv_splits:
        raise ValueError(
            "The training split has too few examples per class for "
            f"{args.cv_splits}-fold CV: {train_class_counts.to_dict()}"
        )

    cv = StratifiedKFold(
        n_splits=args.cv_splits,
        shuffle=True,
        random_state=args.seed,
    )
    search = make_search(
        model_name=args.model,
        seed=args.seed,
        cv=cv,
        scoring=args.scoring,
        n_jobs=n_jobs,
        search_iterations=args.search_iterations,
        n_train_samples=len(train_idx),
        n_features=embeddings.shape[1],
        smoke_test=args.smoke_test,
    )

    started_at = datetime.now()
    print(
        f"Fitting {len(train_idx)} training samples; "
        f"holding out {len(test_idx)} test samples."
    )
    search.fit(embeddings[train_idx], y[train_idx])
    finished_at = datetime.now()

    best_model = search.best_estimator_
    train_metrics = calculate_metrics(
        best_model,
        embeddings[train_idx],
        y[train_idx],
    )
    test_metrics = calculate_metrics(
        best_model,
        embeddings[test_idx],
        y[test_idx],
    )
    test_predictions = best_model.predict(embeddings[test_idx])

    run_name = f"test_size={args.test_size:g}_seed={args.seed}"
    outdir = (
        output_root
        / f"{started_at:%Y%m%d}"
        / "IS_classification"
        / args.classification
        / args.model
        / args.embedding
        / run_name
    )
    outdir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "save_type": "sklearn",
        "best_estimator": best_model,
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "cv_results": pd.DataFrame(search.cv_results_),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "y_train": y[train_idx],
        "y_test": y[test_idx],
        "pred_test": test_predictions,
        "classes": best_model.classes_,
        "embedding_name": args.embedding,
        "embedding_path": str(embedding_path),
        "embedding_dim": int(embeddings.shape[1]),
        "dataset_path": str(dataset_path),
        "classification": args.classification,
        "target_column": target_column,
        "split_path": str(split_path),
        "run_config": vars(args),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
    }

    model_path = outdir / "model_and_results.pkl"
    joblib.dump(artifact, model_path)

    prediction_table = data.iloc[test_idx].copy()
    prediction_table.insert(0, "row_index", test_idx)
    prediction_table["predicted_class"] = test_predictions
    prediction_table.to_csv(outdir / "test_predictions.csv", index=False)

    summary = {
        "model": args.model,
        "classification": args.classification,
        "embedding": args.embedding,
        "embedding_dim": int(embeddings.shape[1]),
        "seed": args.seed,
        "test_size": args.test_size,
        "cv_splits": args.cv_splits,
        "scoring": args.scoring,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "class_counts": pd.Series(y).value_counts().sort_index().to_dict(),
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "dataset_path": str(dataset_path),
        "embedding_path": str(embedding_path),
        "split_path": str(split_path),
        "model_path": str(model_path),
        "started_at": artifact["started_at"],
        "finished_at": artifact["finished_at"],
        "duration_seconds": artifact["duration_seconds"],
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=json_ready),
        encoding="utf-8",
    )

    print(f"Best CV {args.scoring}: {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")
    print(f"Test balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Test macro F1: {test_metrics['f1_macro']:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
