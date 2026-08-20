"""Train simple regressors on precomputed embeddings with fixed splits.

This script is designed to be compatible with the repository's dataset layout
where each dataset lives in a subfolder under `01_DATASETS/` and provides a
*test index* file under that folder.

Key behavior:
- You provide (or we infer) a CSV containing targets.
- You provide (or we infer) a test index file.
- We compute train indices as the complement of the test indices.
- For MLP, a validation split is carved out of train internally.

Repo conventions used by inference:
- Data CSV:     01_DATASETS/<dataset>/*.csv
- Test indices: 01_DATASETS/<dataset>/<split>_split_test_idx.npy
- Embeddings:   02_EMBEDDINGS/<dataset>/<encoder>.npy

Typical repo layout:

01_DATASETS/<dataset>/
  - *.csv
  - <split>_split_test_idx.npy

02_EMBEDDINGS/<dataset>/<encoder>.npy

Example:
  python 05_PY_SCRIPTS/run_model_train.py \
    --dataset CD \
    --embedding molclr \
    --split OB \
    --model linear
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Make `modules` importable regardless of CWD
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from modules.linreg import fit_linear_model_with_fixed_split
from modules.random_forest import fit_random_forest_with_fixed_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train models on frozen embeddings")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=(
            "Dataset identifier. Typically the subfolder name under 01_DATASETS/ "
            "(e.g. CD, HGAS_calc, HOF_exp, HSUB_calc, log(h50)_exp)."
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["linear", "rf", "xgboost", "mlp", "tabpfn"],
        default="linear",
        help="Downstream model to run",
    )

    parser.add_argument(
        "--embedding",
        type=str,
        default="molclr",
        help=(
            "Encoder/embedding name (filename stem) under 02_EMBEDDINGS/<dataset>/ "
            "(e.g. 'molclr' -> 02_EMBEDDINGS/<dataset>/molclr.npy)."
        ),
    )

    # Data / split inputs
    parser.add_argument(
        "--data_csv",
        type=str,
        default=None,
        help="Optional explicit path to dataset CSV. If omitted, inferred from 01_DATASETS/<dataset>.",
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        default=None,
        help=(
            "Optional explicit path to .npy embedding matrix. If omitted, inferred from "
            "02_EMBEDDINGS/<dataset>/<embedding>.npy."
        ),
    )
    parser.add_argument(
        "--test_idx_path",
        type=str,
        default=None,
        help="Optional explicit path to a .npy test index file. If omitted, inferred from split settings.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="random",
        help=(
            "Split identifier (e.g. 'kmf', 'scaffold', 'random'). Used to infer test index file if --test_idx_path is not provided. "
            "Expected file: 01_DATASETS/<dataset>/<split>_split_test_idx.npy."
        ),
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help=(
            "Seed for random splits. Used to infer test index file if --test_idx_path is not provided and split is 'random'. "
        ),
    )

    parser.add_argument(
        "--datasets_dir",
        type=str,
        default="01_DATASETS",
        help="Root folder containing datasets",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="02_EMBEDDINGS",
        help="Root folder containing embeddings",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="07_OUTPUTS",
        help="Root output directory for checkpoints/results",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Model/training seed",
    )

    parser.add_argument(
        "--tabpfn_no_standardize",
        action="store_true",
        help="Disable StandardScaler->TabPFN pipeline wrapping.",
    )
    parser.add_argument(
        "--tabpfn_kwargs_json",
        type=str,
        default=None,
        help=(
            "Optional JSON dict forwarded to TabPFNRegressor constructor. "
            "Example: --tabfm_kwargs_json '{\"n_ensembles\": 8}'."
        ),
    )

    return parser.parse_args()


def _infer_data_csv(datasets_dir: Path, dataset: str) -> Path:
    # Prefer <datasets_dir>/<dataset>/*.csv; fall back to <datasets_dir>/<dataset>.csv
    ds_dir = datasets_dir / dataset
    if ds_dir.exists() and ds_dir.is_dir():
        csvs = sorted(ds_dir.glob("*.csv"))
        if len(csvs) == 1:
            return csvs[0]
        if len(csvs) > 1:
            raise ValueError(
                f"Multiple CSVs found in {ds_dir}. Provide --data_csv explicitly. Found: {csvs}"
            )

    flat = datasets_dir / f"{dataset}.csv"
    if flat.exists():
        return flat

    raise FileNotFoundError(
        f"Could not infer data CSV. Looked in {ds_dir}/*.csv and {flat}. Provide --data_csv."
    )


def _infer_test_idx_path(datasets_dir: Path, dataset: str, split: str, split_seed: int) -> Path:
    """Infer test index file.

    Repo convention:
      01_DATASETS/<dataset>/<split>_split_test_idx.npy
    Backward-compatible fallback:
      01_DATASETS/<dataset>/<split>_split_test.npy
    """

    if Path(split).name != split:
        raise ValueError(
            "--split must be a split identifier (e.g. 'random_seed0'), not a path. "
            "Use --test_idx_path to pass an explicit file path."
        )

    ds_dir = datasets_dir / dataset

    if split.endswith(".npy"):
        return ds_dir / split

    p_primary = ds_dir / f"{split}_split_test_idx.npy"
    if p_primary.exists():
        return p_primary

    p_fallback = ds_dir / f"{split}_split_test.npy"
    return p_fallback


def _infer_embedding_path(embeddings_dir: Path, dataset: str, embedding: str) -> Path:
    """Infer embedding path following repo conventions.

    Preferred:
      02_EMBEDDINGS/<dataset>/<encoder>.npy
    Fallback:
      02_EMBEDDINGS/<dataset>/<encoder>_embeddings.npy
    """
    if Path(embedding).name != embedding:
        raise ValueError(
            "--embedding must be an encoder name (e.g. 'molclr'), not a path. "
            "Use --embedding_path to pass an explicit file path."
        )

    if embedding.endswith(".npy"):
        return embeddings_dir / dataset / embedding

    p_primary = embeddings_dir / dataset / f"{embedding}.npy"
    if p_primary.exists():
        return p_primary

    return embeddings_dir / dataset / f"{embedding}_embeddings.npy"



def _load_indices(test_idx_path: Path, n: int) -> tuple[np.ndarray, np.ndarray]:
    test_idx = np.asarray(np.load(test_idx_path)).astype(int)
    if test_idx.ndim != 1:
        test_idx = test_idx.ravel()

    if len(test_idx) == 0:
        raise ValueError(f"Empty test index array: {test_idx_path}")

    if test_idx.min() < 0 or test_idx.max() >= n:
        raise ValueError(
            f"test_idx out of bounds for dataset size {n}. "
            f"min={test_idx.min()}, max={test_idx.max()}"
        )

    all_idx = np.arange(n, dtype=int)
    train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=False)

    if len(train_idx) == 0:
        raise ValueError("Computed empty train_idx (test covers all samples)")

    return train_idx, test_idx


def run_rf(embd: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, args: argparse.Namespace):
    return fit_random_forest_with_fixed_split(
        embd,
        y,
        train_idx,
        test_idx,
        seed=args.seed,
    )

def run_xgboost(embd: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, args: argparse.Namespace):
    return fit_random_forest_with_fixed_split(
        embd,
        y,
        train_idx,
        test_idx,
        seed=args.seed,
        model_type="xgboost"
    )

def run_linear(embd: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, args: argparse.Namespace):
    return fit_linear_model_with_fixed_split(
        embd,
        y,
        train_idx,
        test_idx,
        seed=args.seed,
        cv_splits=5,
        alphas=np.logspace(-5, 1, 30),
        l1_ratios=(0.0, 1e-4, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.035),
        max_iter=10000,
        scoring="neg_mean_absolute_error",
    )


def run_mlp(embd: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, args: argparse.Namespace):
    try:
        from modules.mlp_tuning import tune_mlp_with_fixed_split
    except ImportError as exc:
        raise RuntimeError(
            "MLP training requires the PyTorch dependencies in environment.yml."
        ) from exc
    # Hyperparameter tuning happens strictly within the TRAIN split via CV.
    # The TEST split is used only once for final evaluation.
    return tune_mlp_with_fixed_split(
        embd,
        y,
        train_idx,
        test_idx,
        seed=args.seed,
        cv_splits=3,
        n_iter=20,
        scoring="neg_mean_absolute_error",
        # per-trial training budget (early stopping typically ends much sooner)
        num_epochs=250,
        patience=20,
        min_delta=1e-5,
        # final refit budget
        refit_num_epochs=400,
        print_every=0,
    )


def run_tabpfn(embd: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, args: argparse.Namespace):
    try:
        from modules.tabpfn import fit_tabpfn_with_fixed_split
    except ImportError as exc:
        raise RuntimeError(
            "TabPFN support is optional and is not included in the anonymized "
            "submission. Use linear, rf, xgboost, or mlp."
        ) from exc
    return fit_tabpfn_with_fixed_split(
        embd,
        y,
        train_idx,
        test_idx,
        seed=args.seed,
    )


if __name__ == "__main__":
    args = parse_args()

    base_dir = Path(os.environ.get("PROJECT_DIR", REPO_ROOT)).resolve()
    datasets_dir = (base_dir / args.datasets_dir).resolve()
    embeddings_dir = (base_dir / args.embeddings_dir).resolve()

    dataset_path = Path(args.data_csv).expanduser().resolve() if args.data_csv else _infer_data_csv(datasets_dir, args.dataset)

    # Embedding path convention:
    #   02_EMBEDDINGS/<dataset>/<encoder>.npy
    # For explicit paths, use --embedding_path.
    embedding_path = (
        Path(args.embedding_path).expanduser().resolve()
        if args.embedding_path
        else _infer_embedding_path(embeddings_dir, args.dataset, args.embedding)
    )

    test_idx_path = (
        Path(args.test_idx_path).expanduser().resolve()
        if args.test_idx_path
        else _infer_test_idx_path(datasets_dir, args.dataset, args.split, args.split_seed)
    )

    print(f"Dataset CSV:     {dataset_path}")
    print(f"Embedding file:  {embedding_path}")
    print(f"Test idx file:   {test_idx_path}")


    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_path}")
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embedding_path}")
    if not test_idx_path.exists():
        raise FileNotFoundError(f"Test split not found: {test_idx_path}")

    data = pd.read_csv(dataset_path)
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns="Unnamed: 0")

    y = data["val"].to_numpy() 

    embd = np.load(embedding_path)
    if embd.ndim == 1:
        embd = embd.reshape(-1, 1)
    if embd.shape[0] != len(y):
        raise ValueError(
            f"Embedding rows ({embd.shape[0]}) do not match dataset rows ({len(y)}). "
            f"Embedding file: {embedding_path}"
        )
    embd_dims = int(embd.shape[1])

    train_idx, test_idx = _load_indices(test_idx_path, n=len(y))

    # Output directory includes split + embedding for reproducibility
    outdir = (
        base_dir
        / args.output_dir
        / f"{datetime.now():%Y%m%d}"
        / args.dataset
        / args.model
        / f"{args.embedding}"
        / f"split={args.split}_modelseed={args.seed}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    MODELS = {
        "rf": run_rf,
        "xgboost": run_xgboost,  # same function, different model_type
        "linear": run_linear,
        "mlp": run_mlp,
        "tabpfn": run_tabpfn,
    }
    
    fit_fn = MODELS[args.model]
    model_dict = fit_fn(embd, y, train_idx, test_idx,  args)

    # Metadata
    model_dict["embedding_name"] = args.embedding
    model_dict["embedding_path"] = str(embedding_path)
    model_dict["dataset"] = args.dataset
    model_dict["dataset_path"] = str(dataset_path)
    model_dict["split"] = args.split
    model_dict["split_seed"] = args.split_seed
    model_dict["test_idx_path"] = str(test_idx_path)
    model_dict["run_config"] = vars(args)
    model_dict["embedding_dim"] = embd_dims
    model_dict["timestamp"] = datetime.now().isoformat()

    # Save
    ckpt_dir = outdir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{args.embedding}_model={args.model}_seed={args.seed}"

    if model_dict.get("save_type") == "sklearn":
        path = ckpt_dir / f"{stem}.pkl"
        try:
            joblib.dump(model_dict, path)
        except Exception as e:
            # Some estimators (or nested objects) may not be picklable in some environments.
            # Fall back to a lightweight JSON that at least captures metrics + provenance.
            print(
                f"[WARN] joblib dump failed ({type(e).__name__}: {e}). "
                "Falling back to JSON without estimator artifacts."
            )
            safe_dict = dict(model_dict)
            for k in ["best_estimator", "cv_results", "pred_train", "pred_test"]:
                safe_dict.pop(k, None)

            path = ckpt_dir / f"{stem}.json"
            path.write_text(json.dumps(safe_dict, indent=2, default=str))
    elif model_dict.get("save_type") == "torch":
        import torch

        path = ckpt_dir / f"{stem}.pt"
        torch.save(model_dict, path)
    else:
        # Fallback: still write something useful
        path = ckpt_dir / f"{stem}.json"
        path.write_text(json.dumps(model_dict, indent=2, default=str))

    # Always write a small JSON summary for quick inspection
    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "embedding": args.embedding,
        "embedding_dim": embd_dims,
        "split": args.split,
        "split_seed": args.split_seed,
        "seed": args.seed,
        "dataset_path": str(dataset_path),
        "embedding_path": str(embedding_path),
        "test_idx_path": str(test_idx_path),
        "outdir": str(outdir),
        "timestamp": model_dict["timestamp"],
        # sklearn baselines use train_metrics/test_metrics; MLP tuner uses train_info/test_metrics
        "train_metrics": (
            model_dict.get("train_metrics")
            or model_dict.get("train")
            or (model_dict.get("train_info") or {}).get("train")
        ),
        "test_metrics": (
            model_dict.get("test_metrics")
            or model_dict.get("test")
            or model_dict.get("test_info")
        ),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"Results saved to: {ckpt_dir}")
    summary_path = outdir / f"summary_seed={args.seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Summary saved to: {summary_path}")
