#!/usr/bin/env python3
"""Nested, leakage-safe Chemprop experiments for the IS dataset.

The saved ``*_split_test.npy`` files are treated as *outer* test splits.  For
each outer split this script:

1. makes deterministic inner K-fold splits from the outer training rows;
2. evaluates the same candidate hyperparameters on every inner fold;
3. selects the candidate with the lowest mean inner validation MAE;
4. retrains one model per inner fold using the selected candidate and averages
   their predictions on the untouched outer test set.

The experiment is repeated for a randomly initialised Chemprop model and for a
model fine-tuned from the CheMeleon foundation model.  Final-model split files
contain absolute row positions in the original IS CSV; HPO split files use
local positions in an outer-training-only CSV.  This prevents accidental
re-splitting, row re-ordering, or outer-test access during optimization.

This is an orchestration script: Chemprop itself is intentionally invoked via
its CLI so that the exact Chemprop version installed in the training
environment controls model construction and checkpointing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ParameterSampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "01_DATASETS" / "IS" / "IS.csv"
DEFAULT_SPLIT_DIR = REPO_ROOT / "01_DATASETS" / "IS"
DEFAULT_OUTPUT = REPO_ROOT / "07_OUTPUTS" / "chemprop_is_cv"
DEFAULT_CHEMELEON_FOUNDATION = REPO_ROOT / "04_FM_CPS" / "chemeleon_mp.pt"
TARGET_COLUMN = "IS (J)"
SMILES_COLUMN = "SMILES"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-safe nested CV and Chemprop/CheMeleon comparison for IS."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument(
        "--outer-splits",
        nargs="+",
        default=None,
        help=(
            "Split names or .npy paths. Defaults to all random_seed*_split_test.npy "
            "files in --split-dir."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--chemeleon-foundation",
        type=Path,
        default=DEFAULT_CHEMELEON_FOUNDATION,
        help=(
            "Local CheMeleon foundation checkpoint. Using a local file avoids "
            "Chemprop's home-directory foundation cache."
        ),
    )
    parser.add_argument("--chemprop-command", default="chemprop")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("scratch", "chemeleon"),
        default=("scratch", "chemeleon"),
        help="Model initialisations to compare.",
    )
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument(
        "--hpo-trials",
        type=int,
        default=12,
        help="Number of deterministic candidate configurations per outer split and mode.",
    )
    parser.add_argument("--hpo-epochs", type=int, default=40)
    parser.add_argument("--hpo-patience", type=int, default=12)
    parser.add_argument("--final-epochs", type=int, default=100)
    parser.add_argument("--final-patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--accelerator",
        default="auto",
        help="Chemprop Lightning accelerator, e.g. auto, cpu, or gpu.",
    )
    parser.add_argument(
        "--devices",
        default="auto",
        help="Chemprop Lightning devices, e.g. auto, 0, or 0,1.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run completed candidate/final runs instead of resuming them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data/splits and print the first commands without running Chemprop.",
    )
    return parser.parse_args()


def resolve_path(path: Path, base: Path = REPO_ROOT) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def jsonable(value: Any) -> Any:
    """Convert numpy scalar values to ordinary JSON values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def candidate_id(candidate: dict[str, Any]) -> str:
    payload = json.dumps(candidate, sort_keys=True, separators=(",", ":"), default=jsonable)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    data = pd.read_csv(data_path)
    missing = {SMILES_COLUMN, TARGET_COLUMN} - set(data.columns)
    if missing:
        raise KeyError(f"Missing required IS columns {sorted(missing)} in {data_path}")
    if data[[SMILES_COLUMN, TARGET_COLUMN]].isna().any().any():
        raise ValueError(f"{data_path} contains missing SMILES or {TARGET_COLUMN} values")
    smiles = data[SMILES_COLUMN].astype(str).to_numpy()
    y = pd.to_numeric(data[TARGET_COLUMN], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError(f"{TARGET_COLUMN} contains non-finite values")
    if data[SMILES_COLUMN].astype(str).duplicated().any():
        raise ValueError("Duplicate raw SMILES detected; refusing to run a leakage-sensitive benchmark")
    if "canon_SMILES" in data.columns and data["canon_SMILES"].astype(str).duplicated().any():
        raise ValueError(
            "Duplicate canonical SMILES detected; refusing to run because a fixed split "
            "could place the same molecule in train and test."
        )
    return data, y


def resolve_split_path(value: str, split_dir: Path) -> Path:
    raw = Path(value).expanduser()
    candidates: list[Path]
    if raw.suffix == ".npy" or raw.parent != Path("."):
        candidates = [resolve_path(raw, REPO_ROOT)]
    else:
        name = raw.name
        candidates = [
            split_dir / name,
            split_dir / f"{name}_split_test_idx.npy",
            split_dir / f"{name}_split_test.npy",
        ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Could not find saved test-index file for {value!r}: {candidates}")


def load_test_indices(path: Path, n_rows: int) -> np.ndarray:
    indices = np.asarray(np.load(path))
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"Test indices must be a one-dimensional integer array: {path}")
    indices = indices.astype(int, copy=False)
    if len(indices) == 0 or len(np.unique(indices)) != len(indices):
        raise ValueError(f"Test indices are empty or duplicated: {path}")
    if indices.min() < 0 or indices.max() >= n_rows:
        raise IndexError(f"Test indices in {path} are outside [0, {n_rows - 1}]")
    return indices


def validate_outer_split(data: pd.DataFrame, test_idx: np.ndarray, split_path: Path) -> np.ndarray:
    all_idx = np.arange(len(data), dtype=int)
    train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)
    if not len(train_idx):
        raise ValueError(f"Empty training complement for {split_path}")
    if "canon_SMILES" in data.columns:
        canonical = data["canon_SMILES"].astype(str).to_numpy()
        if set(canonical[train_idx]) & set(canonical[test_idx]):
            raise ValueError(f"Canonical molecule overlap between train/test for {split_path}")
    return train_idx


def make_candidates(mode: str, n_trials: int, seed: int) -> list[dict[str, Any]]:
    """Make a reproducible, explicit search space.

    CheMeleon fixes the message-passing representation, so its search omits
    message-passing dimensions that Chemprop ignores when ``--from-foundation``
    is supplied.
    """
    common = {
        "ffn_hidden_dim": [100, 200, 300, 500, 800],
        "ffn_num_layers": [1, 2, 3],
        "dropout": [0.0, 0.05, 0.1, 0.2, 0.3],
        "max_lr": [1e-4, 3e-4, 1e-3, 3e-3],
        "init_lr": [1e-5, 3e-5, 1e-4],
        "final_lr": [1e-6, 1e-5, 3e-5],
        "warmup_epochs": [1, 2, 4, 6],
    }
    if mode == "scratch":
        common.update(
            {
                "message_hidden_dim": [300, 600, 1200],
                "depth": [3, 4, 5],
                "aggregation": ["mean", "norm"],
            }
        )
    candidates = list(ParameterSampler(common, n_iter=n_trials, random_state=seed))
    return [{key: jsonable(value) for key, value in candidate.items()} for candidate in candidates]


def write_split(path: Path, train: np.ndarray, val: np.ndarray, test: np.ndarray | None = None) -> None:
    split: dict[str, list[int]] = {"train": train.astype(int).tolist(), "val": val.astype(int).tolist()}
    if test is not None:
        split["test"] = test.astype(int).tolist()
    path.write_text(json.dumps([split], indent=2) + "\n", encoding="utf-8")


def base_command(
    args: argparse.Namespace,
    output_dir: Path,
    split_file: Path,
    data_path: Path | None = None,
) -> list[str]:
    data_path = data_path or resolve_path(args.data_path)
    return [
        args.chemprop_command,
        "train",
        "--data-path",
        str(data_path),
        "--output-dir",
        str(output_dir),
        "--splits-file",
        str(split_file),
        "--task-type",
        "regression",
        "--smiles-columns",
        SMILES_COLUMN,
        "--target-columns",
        TARGET_COLUMN,
        "--metrics",
        "mae",
        "rmse",
        "r2",
        "--tracking-metric",
        "mae",
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--accelerator",
        args.accelerator,
        "--devices",
        args.devices,
    ]


def add_candidate_args(command: list[str], candidate: dict[str, Any]) -> list[str]:
    option_names = {
        "message_hidden_dim": "--message-hidden-dim",
        "ffn_hidden_dim": "--ffn-hidden-dim",
        "ffn_num_layers": "--ffn-num-layers",
        "warmup_epochs": "--warmup-epochs",
        "init_lr": "--init-lr",
        "max_lr": "--max-lr",
        "final_lr": "--final-lr",
        "dropout": "--dropout",
        "depth": "--depth",
        "aggregation": "--aggregation",
    }
    for key, value in candidate.items():
        command.extend([option_names[key], str(value)])
    return command


def run_command(command: list[str], log_path: Path, dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        if dry_run:
            return
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
    if result.returncode:
        raise RuntimeError(
            f"Chemprop command failed with exit code {result.returncode}. See {log_path}"
        )


def validation_score(run_dir: Path) -> float:
    """Read the best validation MAE from Chemprop's Lightning logs.

    Chemprop selects ``TensorBoardLogger`` when TensorBoard is installed and
    otherwise falls back to ``CSVLogger``.  The Eddie environment contains
    TensorBoard, so both formats must be supported here.
    """

    def score_from_columns(frame: pd.DataFrame) -> float | None:
        preferred: list[tuple[int, str]] = []
        for column in frame.columns:
            low = str(column).lower()
            if "val" in low and "mae" in low:
                preferred.append((0, column))
            elif low in {"val_loss", "val/loss", "val-loss"}:
                preferred.append((1, column))
        for _, column in sorted(preferred, key=lambda x: x[0]):
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if len(values):
                return float(values.min())
        return None

    metric_files = sorted(run_dir.rglob("metrics.csv"))
    for path in metric_files:
        frame = pd.read_csv(path)
        score = score_from_columns(frame)
        if score is not None:
            return score

    event_files = sorted(run_dir.rglob("events.out.tfevents.*"))
    if event_files:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

            for path in event_files:
                accumulator = EventAccumulator(str(path))
                accumulator.Reload()
                preferred_tags: list[tuple[int, str]] = []
                for tag in accumulator.Tags().get("scalars", []):
                    low = tag.lower()
                    if "val" in low and "mae" in low:
                        preferred_tags.append((0, tag))
                    elif low in {"val_loss", "val/loss", "val-loss"}:
                        preferred_tags.append((1, tag))
                for _, tag in sorted(preferred_tags, key=lambda x: x[0]):
                    values = [event.value for event in accumulator.Scalars(tag)]
                    if values:
                        return float(np.min(values))
        except Exception as exc:
            tensorboard_error = f"TensorBoard parsing failed: {exc}"
        else:
            tensorboard_error = None
    else:
        tensorboard_error = None

    # Last-resort support for a logger configuration that only emits scalar
    # values to stdout/stderr.
    log_path = run_dir / "command.log"
    if log_path.is_file():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        matches = re.findall(
            r"(?:val[/_-](?:mae|loss)|val(?:[/_-])?mae)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            text,
            flags=re.IGNORECASE,
        )
        if matches:
            return float(np.min(np.asarray(matches, dtype=float)))

    details = f"CSV files: {metric_files}; TensorBoard files: {event_files}"
    if tensorboard_error:
        details += f"; {tensorboard_error}"
    raise RuntimeError(
        f"Could not find a validation metric in {run_dir}. {details}"
    )


def read_test_predictions(run_dir: Path) -> np.ndarray:
    files = sorted(run_dir.rglob("test_predictions.csv"))
    if not files:
        raise RuntimeError(f"Chemprop did not write test_predictions.csv below {run_dir}")
    frame = pd.read_csv(files[0])
    # Chemprop 2.2 writes input + target + prediction columns.  The prediction
    # is the final column; selecting by position also handles duplicate target
    # column names in that file.
    values = pd.to_numeric(frame.iloc[:, -1], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite predictions in {files[0]}")
    return values


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def fit_one_mode_one_split(
    args: argparse.Namespace,
    data: pd.DataFrame,
    y: np.ndarray,
    split_name: str,
    test_idx: np.ndarray,
    train_idx: np.ndarray,
    mode: str,
    split_number: int,
    chemeleon_foundation: Path | None,
) -> dict[str, Any]:
    split_root = resolve_path(args.output_dir) / mode / split_name
    hpo_root = split_root / "hpo"
    final_root = split_root / "final_cv_ensemble"
    hpo_root.mkdir(parents=True, exist_ok=True)
    final_root.mkdir(parents=True, exist_ok=True)

    inner_seed = args.seed + 1009 * split_number
    kfold = KFold(n_splits=args.inner_folds, shuffle=True, random_state=inner_seed)
    local_folds = list(kfold.split(np.arange(len(train_idx), dtype=int)))
    folds = [(train_idx[tr], train_idx[va]) for tr, va in local_folds]
    candidates = make_candidates(mode, args.hpo_trials, inner_seed)

    # HPO receives a file containing only the outer-training rows.  Its split
    # files therefore use local positions, making it impossible for an HPO
    # command to inspect an outer-test molecule, even accidentally.
    hpo_data_path = split_root / "outer_train_pool.csv"
    if not hpo_data_path.exists() or args.force:
        data.iloc[train_idx].to_csv(hpo_data_path, index=False)

    for fold_number, (fold_train_local, fold_val_local) in enumerate(local_folds):
        fold_file = hpo_root / f"fold_{fold_number}_splits.json"
        if not fold_file.exists() or args.force:
            write_split(fold_file, fold_train_local, fold_val_local)

    rows: list[dict[str, Any]] = []
    for trial_number, candidate in enumerate(candidates):
        cid = candidate_id(candidate)
        for fold_number, _ in enumerate(folds):
            score_file = hpo_root / f"trial_{trial_number:03d}_{cid}" / f"fold_{fold_number}" / "score.json"
            if score_file.exists() and not args.force:
                score = json.loads(score_file.read_text(encoding="utf-8"))["val_mae"]
            else:
                run_dir = score_file.parent
                split_file = hpo_root / f"fold_{fold_number}_splits.json"
                command = base_command(args, run_dir, split_file, hpo_data_path)
                command = add_candidate_args(command, candidate)
                command.extend(
                    [
                        "--epochs",
                        str(args.hpo_epochs),
                        "--patience",
                        str(args.hpo_patience),
                        "--pytorch-seed",
                        str(inner_seed + trial_number),
                        "--remove-checkpoints",
                    ]
                )
                if mode == "chemeleon":
                    command.extend(["--from-foundation", str(chemeleon_foundation)])
                run_command(command, run_dir / "command.log", args.dry_run)
                if args.dry_run:
                    score = float("nan")
                else:
                    score = validation_score(run_dir)
                    score_file.write_text(
                        json.dumps({"val_mae": score, "candidate": candidate}, indent=2) + "\n",
                        encoding="utf-8",
                    )
            rows.append(
                {
                    "mode": mode,
                    "split": split_name,
                    "trial": trial_number,
                    "candidate_id": cid,
                    "fold": fold_number,
                    "val_mae": score,
                    **candidate,
                }
            )
            pd.DataFrame(rows).to_csv(split_root / "inner_cv_results.csv", index=False)

    if args.dry_run:
        return {"mode": mode, "split": split_name, "dry_run": True}

    cv_frame = pd.DataFrame(rows)
    grouped = (
        cv_frame.groupby(["trial", "candidate_id"], as_index=False)
        .agg(mean_val_mae=("val_mae", "mean"), std_val_mae=("val_mae", "std"), n_folds=("val_mae", "count"))
        .sort_values(["mean_val_mae", "std_val_mae", "trial"])
    )
    if grouped.empty or grouped.iloc[0]["n_folds"] != len(folds):
        raise RuntimeError(f"Incomplete inner CV results for {mode}/{split_name}")
    best_trial = int(grouped.iloc[0]["trial"])
    best_candidate = candidates[best_trial]
    best_id = candidate_id(best_candidate)
    grouped.to_csv(split_root / "inner_cv_summary.csv", index=False)
    (split_root / "best_hyperparameters.json").write_text(
        json.dumps(
            {
                "selection_rule": "minimum mean inner validation MAE across disjoint K folds",
                "best_trial": best_trial,
                "candidate_id": best_id,
                "candidate": best_candidate,
                "inner_seed": inner_seed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    fold_predictions: list[np.ndarray] = []
    for fold_number, (fold_train, fold_val) in enumerate(folds):
        run_dir = final_root / f"fold_{fold_number}"
        prediction_file = run_dir / "predictions.npy"
        if prediction_file.exists() and not args.force:
            prediction = np.load(prediction_file)
        else:
            split_file = final_root / f"fold_{fold_number}_splits.json"
            if not split_file.exists() or args.force:
                write_split(split_file, fold_train, fold_val, test_idx)
            command = base_command(args, run_dir, split_file)
            command = add_candidate_args(command, best_candidate)
            command.extend(
                [
                    "--epochs",
                    str(args.final_epochs),
                    "--patience",
                    str(args.final_patience),
                    "--pytorch-seed",
                    str(args.seed + split_number * 100 + fold_number),
                ]
            )
            if mode == "chemeleon":
                command.extend(["--from-foundation", str(chemeleon_foundation)])
            run_command(command, run_dir / "command.log", args.dry_run)
            prediction = read_test_predictions(run_dir)
            np.save(prediction_file, prediction)
        if len(prediction) != len(test_idx):
            raise ValueError(
                f"Expected {len(test_idx)} test predictions, got {len(prediction)} for {run_dir}"
            )
        fold_predictions.append(np.asarray(prediction, dtype=float))

    prediction = np.mean(np.stack(fold_predictions, axis=0), axis=0)
    test_metrics = metrics(y[test_idx], prediction)
    prediction_frame = pd.DataFrame(
        {
            "row_index": test_idx,
            SMILES_COLUMN: data.iloc[test_idx][SMILES_COLUMN].astype(str).to_numpy(),
            "y_true": y[test_idx],
            "y_pred": prediction,
        }
    )
    prediction_frame.to_csv(split_root / "outer_test_predictions.csv", index=False)
    result = {
        "mode": mode,
        "split": split_name,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "inner_folds": int(args.inner_folds),
        "hpo_trials": int(args.hpo_trials),
        "best_trial": best_trial,
        "best_candidate_id": best_id,
        "best_candidate": best_candidate,
        "inner_cv_mean_mae": float(grouped.iloc[0]["mean_val_mae"]),
        "inner_cv_std_mae": float(grouped.iloc[0]["std_val_mae"]),
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    (split_root / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    args = parse_args()
    if args.inner_folds < 2 or args.hpo_trials < 1:
        raise ValueError("--inner-folds must be >= 2 and --hpo-trials must be >= 1")
    if args.data_path.name != "IS.csv" and "IS" not in args.data_path.parts:
        raise ValueError("This script is intentionally restricted to the IS dataset")

    data_path = resolve_path(args.data_path)
    split_dir = resolve_path(args.split_dir)
    data, y = load_dataset(data_path)
    if args.outer_splits is None:
        split_paths = sorted(split_dir.glob("random_seed*_split_test_idx.npy"))
        split_paths += sorted(split_dir.glob("random_seed*_split_test.npy"))
        # Prefer *_split_test_idx.npy if both naming conventions are present.
        by_name: dict[str, Path] = {}
        for path in split_paths:
            split_name = path.stem.replace("_split_test_idx", "").replace("_split_test", "")
            by_name.setdefault(split_name, path.resolve())
        split_paths = [by_name[name] for name in sorted(by_name)]
    else:
        split_paths = [resolve_split_path(value, split_dir) for value in args.outer_splits]
    if not split_paths:
        raise FileNotFoundError(f"No random_seed*_split_test(.npy|_idx.npy) files in {split_dir}")

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chemeleon_foundation = resolve_path(args.chemeleon_foundation)
    if "chemeleon" in args.modes and not chemeleon_foundation.is_file():
        raise FileNotFoundError(
            "CheMeleon foundation checkpoint not found: "
            f"{chemeleon_foundation}. Pass --chemeleon-foundation explicitly."
        )
    metadata = {
        "data_path": str(data_path),
        "n_rows": len(data),
        "target_column": TARGET_COLUMN,
        "chemeleon_foundation": str(chemeleon_foundation),
        "outer_split_paths": [str(path) for path in split_paths],
        "modes": args.modes,
        "inner_folds": args.inner_folds,
        "hpo_trials": args.hpo_trials,
        "seed": args.seed,
        "leakage_controls": [
            "outer test indices are loaded from saved .npy files",
            "inner folds are generated only from the complement of each outer test",
            "outer test rows are not passed to any HPO command",
            "canonical SMILES overlap is rejected",
        ],
    }
    (output_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    all_results: list[dict[str, Any]] = []
    for split_number, split_path in enumerate(split_paths):
        split_name = split_path.stem.replace("_split_test_idx", "").replace("_split_test", "")
        test_idx = load_test_indices(split_path, len(data))
        train_idx = validate_outer_split(data, test_idx, split_path)
        for mode in args.modes:
            print(f"\n=== {mode} / {split_name}: train={len(train_idx)} test={len(test_idx)} ===", flush=True)
            result = fit_one_mode_one_split(
                args,
                data,
                y,
                split_name,
                test_idx,
                train_idx,
                mode,
                split_number,
                chemeleon_foundation,
            )
            all_results.append(result)

    if not args.dry_run:
        summary = pd.DataFrame(all_results)
        summary.to_csv(output_dir / "outer_test_summary.csv", index=False)
        aggregate = (
            summary.groupby("mode", as_index=False)
            .agg(
                mean_test_MAE=("test_MAE", "mean"),
                std_test_MAE=("test_MAE", "std"),
                mean_test_RMSE=("test_RMSE", "mean"),
                std_test_RMSE=("test_RMSE", "std"),
                mean_test_R2=("test_R2", "mean"),
                std_test_R2=("test_R2", "std"),
            )
        )
        aggregate.to_csv(output_dir / "outer_test_aggregate.csv", index=False)
        print(f"\nWrote results to {output_dir}")
        print(aggregate.to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
