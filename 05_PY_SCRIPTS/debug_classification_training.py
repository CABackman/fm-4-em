"""Fast compatibility checks for the classification training workflow.

By default this performs no model fitting. Pass ``--fit`` to run one tiny
end-to-end logistic-regression search after all structural checks pass.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION_DIR = REPO_ROOT / "08_CLASSIFICATION"
DATASET_PATH = CLASSIFICATION_DIR / "01_DATASETS" / "combined_IS_dataset.csv"
EMBEDDING_DIR = CLASSIFICATION_DIR / "02_EMBEDDINGS"
RUNNER = REPO_ROOT / "05_PY_SCRIPTS" / "run_classification_model_train.py"
PIPELINES = REPO_ROOT / "modules" / "classification_model_pipelines.py"
LAUNCHER = REPO_ROOT / "03_SH_SCRIPTS" / "train_all_classification.sh"

CLASSIFICATIONS = ("binary", "tertiary", "quaternary", "quinary")
EMBEDDINGS = (
    "hq_descriptor",
    "chemeleon",
    "morgan_fps",
    "maceoff_large",
    "molclr",
)
MODELS = ("svc", "logreg", "rf", "lightgbm", "bayes", "xgboost")
SEEDS = range(5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit",
        action="store_true",
        help="Also run one minimal end-to-end fit (2-fold CV, one candidate).",
    )
    return parser.parse_args()


def check_source_files(errors: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(PIPELINES), str(RUNNER)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        errors.append(f"Python syntax check failed:\n{result.stderr.strip()}")

    result = subprocess.run(
        ["bash", "-n", str(LAUNCHER)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        errors.append(f"Shell syntax check failed:\n{result.stderr.strip()}")


def check_optional_dependencies(errors: list[str]) -> None:
    for package in ("lightgbm", "xgboost"):
        if importlib.util.find_spec(package) is None:
            errors.append(
                f"Optional dependency {package!r} is missing, but the SGE launcher "
                f"schedules its model."
            )


def check_data(errors: list[str]) -> None:
    if not DATASET_PATH.is_file():
        errors.append(f"Missing dataset: {DATASET_PATH}")
        return

    data = pd.read_csv(DATASET_PATH)
    n_samples = len(data)
    expected_test_rows = int(np.ceil(0.20 * n_samples))

    for classification in CLASSIFICATIONS:
        candidates = (classification, f"{classification}_class")
        target = next((name for name in candidates if name in data.columns), None)
        if target is None:
            errors.append(
                f"{classification}: neither target column {candidates[0]!r} nor "
                f"{candidates[1]!r} exists."
            )
            continue
        if data[target].isna().any():
            errors.append(f"{target}: contains {int(data[target].isna().sum())} NaNs.")

        for seed in SEEDS:
            split_path = (
                CLASSIFICATION_DIR
                / "01_DATASETS"
                / classification
                / f"random_seed{seed}_split_test.npy"
            )
            if not split_path.is_file():
                errors.append(f"Missing split: {split_path}")
                continue
            indices = np.asarray(np.load(split_path))
            prefix = f"{classification}/seed{seed}"
            if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
                errors.append(f"{prefix}: split must be a 1D integer array.")
                continue
            if len(indices) != expected_test_rows:
                errors.append(
                    f"{prefix}: has {len(indices)} test rows; expected "
                    f"{expected_test_rows} for the current {n_samples}-row dataset."
                )
            if len(np.unique(indices)) != len(indices):
                errors.append(f"{prefix}: split contains duplicate indices.")
            invalid = indices[(indices < 0) | (indices >= n_samples)]
            if invalid.size:
                errors.append(
                    f"{prefix}: out-of-range indices {np.unique(invalid).tolist()} "
                    f"for {n_samples} rows."
                )
            elif indices.size:
                full_distribution = data[target].value_counts(normalize=True)
                test_distribution = data.iloc[indices][target].value_counts(normalize=True)
                difference = full_distribution.sub(test_distribution, fill_value=0).abs()
                tolerance = 1 / len(indices) + 1 / n_samples
                if (difference > tolerance).any():
                    errors.append(
                        f"{prefix}: test class proportions are not stratified within "
                        f"the expected rounding tolerance ({difference.to_dict()})."
                    )

    for embedding in EMBEDDINGS:
        path = EMBEDDING_DIR / f"{embedding}_embeddings.npy"
        if not path.is_file():
            errors.append(f"Missing embedding: {path}")
            continue
        array = np.load(path, mmap_mode="r")
        if array.ndim != 2:
            errors.append(f"{embedding}: expected 2D data, got {array.shape}.")
        elif array.shape[0] != n_samples:
            errors.append(
                f"{embedding}: has {array.shape[0]} rows; dataset has {n_samples}."
            )
        if not np.issubdtype(array.dtype, np.number):
            errors.append(f"{embedding}: nonnumeric dtype {array.dtype}.")
        elif not np.isfinite(array).all():
            errors.append(f"{embedding}: contains NaN or infinite values.")


def check_task_mapping(errors: list[str]) -> None:
    expected_combinations = {
        (classification, embedding, model, seed)
        for seed in SEEDS
        for model in MODELS
        for embedding in EMBEDDINGS
        for classification in CLASSIFICATIONS
    }
    actual_combinations: list[tuple[str, str, str, int]] = []
    environment = os.environ.copy()
    environment.update(
        {
            "CLASSIFICATION_MAP_ONLY": "1",
            "JOB_ID": "debug",
            "NSLOTS": "1",
        }
    )
    for task_id in range(1, 601):
        environment["SGE_TASK_ID"] = str(task_id)
        result = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            errors.append(
                f"SGE task {task_id} mapping failed: {result.stderr.strip()}"
            )
            return
        classification, model, embedding, seed = result.stdout.splitlines()[-1].split(",")
        actual_combinations.append((classification, embedding, model, int(seed)))

    actual_set = set(actual_combinations)
    if len(actual_combinations) != 600 or len(actual_set) != 600:
        errors.append("SGE launcher does not map to 600 unique task combinations.")
    if actual_set != expected_combinations:
        errors.append("SGE launcher task mapping differs from the expected dimensions.")


def run_minimal_fit() -> None:
    command = [
        sys.executable,
        str(RUNNER),
        "--classification",
        "binary",
        "--model",
        "logreg",
        "--embedding",
        "hq_descriptor",
        "--seed",
        "4",
        "--cv_splits",
        "2",
        "--search_iterations",
        "1",
        "--n_jobs",
        "1",
        "--smoke_test",
        "--output_dir",
        "/tmp/fm-4-em-classification-smoke",
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    check_source_files(errors)
    check_optional_dependencies(errors)
    check_data(errors)
    check_task_mapping(errors)

    if errors:
        print(f"FAILED: found {len(errors)} compatibility issue(s):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        raise SystemExit(1)

    print("PASS: source, dependencies, data, splits, embeddings, and 600-task mapping.")
    if args.fit:
        run_minimal_fit()
        print("PASS: minimal end-to-end fit completed.")


if __name__ == "__main__":
    main()
