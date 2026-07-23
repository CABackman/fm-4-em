from __future__ import annotations

"""Hyperparameter tuning for the VanillaMLP probe.

training + tuning wrapper analogous to the sklearn
baselines in :mod:`modules.linreg` and :mod:`modules.random_forest`.

Design goals
------------
- Never use the TEST split for model selection.
- Perform hyperparameter search using CV on the TRAIN indices only.
- Refit a final model using the best hyperparameters, then evaluate on TEST.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sklearn
import torch
from sklearn.model_selection import KFold, ParameterSampler
import sys
from pathlib import Path
repo_root = Path.cwd().parent.resolve()          # repo root
sys.path.insert(0, str(repo_root))

from modules.basicmlp import (
    VanillaMLP,
    eval_metrics,
    get_tvt,
    seed_all,
    split_train_into_train_val,
    train_with_early_stopping,
)


def _normalize_scoring(scoring: str) -> str:
    """Normalize metric name to {"mae","mse"}."""

    s = scoring.strip().lower()
    if s in {"mae", "l1", "mean_absolute_error"}:
        return "mae"
    if s in {"mse", "l2", "mean_squared_error"}:
        return "mse"

    # Common sklearn scoring strings
    if s in {"neg_mean_absolute_error", "neg_mae"}:
        return "mae"
    if s in {"neg_mean_squared_error", "neg_mse"}:
        return "mse"

    raise ValueError(
        f"Unsupported scoring={scoring!r}. Use 'mae' or 'mse' "
        "(or sklearn-style 'neg_mean_absolute_error' / 'neg_mean_squared_error')."
    )


def _default_param_distributions() -> Dict[str, Iterable[Any]]:
    """Default random-search space.
    """

    return {
        # Optimization
        "lr": [3e-4, 1e-3, 3e-3],
        "weight_decay": [0.0, 1e-6, 1e-5, 1e-4, 1e-3],
        # Architecture
        "proj_dim": [128, 256, 512],
        "hidden1": [128, 256, 512],
        "hidden2": [64, 128, 256],
        # Regularization
        "dropout": [0.0, 0.05, 0.1, 0.2],
    }


@torch.no_grad()
def _metric_from_eval(metrics: Dict[str, float], *, scoring: str) -> float:
    scoring = _normalize_scoring(scoring)
    if scoring == "mae":
        return float(metrics["MAE"])
    if scoring == "mse":
        return float(metrics["MSE"])
    raise RuntimeError("unreachable")


def tune_mlp_with_fixed_split(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    seed: int = 42,
    cv_splits: int = 3,
    scoring: str = "neg_mean_absolute_error",
    n_iter: int = 20,
    param_distributions: Optional[Dict[str, Iterable[Any]]] = None,
    # Training budget per CV fit
    num_epochs: int = 250,
    patience: int = 20,
    min_delta: float = 1e-5,
    # Final refit budget
    refit_num_epochs: int = 400,
    # Device / logging
    device: str | None = None,
    print_every: int = 0,
) -> dict:
    """Hyperparameter tuning for :class:`modules.basicmlp.VanillaMLP`.

    Parameters
    ----------
    X, y : array-like
        Frozen embeddings and targets.
    train_idx, test_idx : np.ndarray
        Fixed outer TRAIN/TEST split.
    seed : int
        Random seed.
    cv_splits : int
        Number of folds for CV on the outer TRAIN indices.
    scoring : str
        Metric to minimize. Supports 'mae'/'mse' and sklearn-style
        'neg_mean_absolute_error' / 'neg_mean_squared_error'.
    n_iter : int
        Number of randomly sampled hyperparameter configurations.
    param_distributions : dict or None
        Search space. If None, a default space is used.
    num_epochs, patience, min_delta :
        Training + early stopping parameters for each CV fit.
    refit_num_epochs : int
        Training budget for the final refit with best hyperparameters.
    device : str or None
        'cpu' or 'cuda'. If None, auto-detect.
    print_every : int
        Passed to the training loop. Use 0 to silence per-epoch logs during CV.

    Returns
    -------
    dict
        Contains best hyperparameters, CV results, and a final refit checkpoint
        evaluated once on the outer TEST split.
    """

    # Basic validation
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)

    if X.shape[0] != len(y):
        raise ValueError(f"X has {X.shape[0]} rows but y has {len(y)}")

    scoring_norm = _normalize_scoring(scoring)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    # CV on TRAIN only
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    if param_distributions is None:
        param_distributions = _default_param_distributions()

    sampler = ParameterSampler(param_distributions, n_iter=n_iter, random_state=seed)

    trial_results: list[dict[str, Any]] = []
    best_score = float("inf")
    best_params: Optional[Dict[str, Any]] = None

    for trial_id, params in enumerate(sampler):
        fold_scores: list[float] = []

        for fold_id, (tr_sub, va_sub) in enumerate(cv.split(train_idx)):
            fold_train_idx = train_idx[tr_sub]
            fold_val_idx = train_idx[va_sub]

            # Deterministic init per (trial, fold)
            seed_all(int(seed + 1000 * trial_id + fold_id))

            # Build TVT. For CV we don't need TEST, but get_tvt requires it.
            # We pass fold_val_idx as a harmless placeholder test split.
            tvt, _scaler = get_tvt(X, y, fold_train_idx, fold_val_idx, fold_val_idx)

            X_train = tvt["X_train"].to(device_t)
            y_train = tvt["y_train"].to(device_t)
            X_val = tvt["X_val"].to(device_t)
            y_val = tvt["y_val"].to(device_t)

            # Scale y to [0, 1] range. 
            y_scaler = MinMaxScaler()

            y_train_np = y_train.detach().cpu().numpy()          # (N,1)
            y_val_np   = y_val.detach().cpu().numpy()            # (M,1)

            y_train_scaled_np = y_scaler.fit_transform(y_train_np)   # (N,1)
            y_val_scaled_np   = y_scaler.transform(y_val_np)         # (M,1)

            y_train_scaled = torch.from_numpy(y_train_scaled_np).to(device_t, dtype=torch.float32)
            y_val_scaled   = torch.from_numpy(y_val_scaled_np).to(device_t, dtype=torch.float32)

            model = VanillaMLP(
                input_size=int(X_train.shape[1]),
                hidden1=int(params["hidden1"]),
                hidden2=int(params["hidden2"]),
                dropout=float(params["dropout"]),
                proj_dim=int(params["proj_dim"]),
            ).to(device=device_t, dtype=torch.float32)

            # train_with_early_stopping currently expects (X_test, y_test).
            # For CV, passes the validation tensor as placeholders. Not
            # used for selection beyond the returned 'val' metrics.
            info = train_with_early_stopping(
                model,
                X_train,
                y_train_scaled,
                X_val,
                y_val_scaled,
                X_val,
                y_val_scaled,
                y_scaler = y_scaler,
                lr=float(params["lr"]),
                weight_decay=float(params["weight_decay"]),
                num_epochs=int(num_epochs),
                patience=int(patience),
                min_delta=float(min_delta),
                print_every=int(print_every),
                early_stop_on=scoring_norm,
            )

            fold_score = _metric_from_eval(info["val"], scoring=scoring_norm)
            fold_scores.append(float(fold_score))

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))

        trial_results.append(
            {
                "trial_id": int(trial_id),
                "params": dict(params),
                "fold_scores": fold_scores,
                "mean_score": mean_score,
                "std_score": std_score,
            }
        )

        if mean_score < best_score:
            best_score = mean_score
            best_params = dict(params)

    if best_params is None:
        raise RuntimeError("No trials were run; check n_iter and param_distributions.")

    # Final refit: split outer TRAIN into a refit train/val (still no TEST leakage)
    refit_train_idx, refit_val_idx = split_train_into_train_val(
        train_idx,
        val_ratio=0.2,
        random_state=seed,
        shuffle=True,
    )

    seed_all(seed)
    tvt, scaler = get_tvt(X, y, refit_train_idx, refit_val_idx, test_idx)

    X_train = tvt["X_train"].to(device_t)
    y_train = tvt["y_train"].to(device_t)
    X_val = tvt["X_val"].to(device_t)
    y_val = tvt["y_val"].to(device_t)
    X_test = tvt["X_test"].to(device_t)
    y_test = tvt["y_test"].to(device_t)

    final_y_scaler = MinMaxScaler()

    y_train_np = y_train.detach().cpu().numpy()  # (N,1)
    y_val_np   = y_val.detach().cpu().numpy()    # (M,1)
    y_test_np  = y_test.detach().cpu().numpy()   # (K,1)

    y_train_scaled_np = final_y_scaler.fit_transform(y_train_np)  # (N,1)
    y_val_scaled_np   = final_y_scaler.transform(y_val_np)        # (M,1)
    y_test_scaled_np  = final_y_scaler.transform(y_test_np)       # (K,1)

    y_train_scaled = torch.from_numpy(y_train_scaled_np).to(device_t, dtype=torch.float32)
    y_val_scaled   = torch.from_numpy(y_val_scaled_np).to(device_t, dtype=torch.float32)
    y_test_scaled  = torch.from_numpy(y_test_scaled_np).to(device_t, dtype=torch.float32)

    best_model = VanillaMLP(
        input_size=int(X_train.shape[1]),
        hidden1=int(best_params["hidden1"]),
        hidden2=int(best_params["hidden2"]),
        dropout=float(best_params["dropout"]),
        proj_dim=int(best_params["proj_dim"]),
    ).to(device=device_t, dtype=torch.float32)

    train_info = train_with_early_stopping(
        best_model,
        X_train,
        y_train_scaled,
        X_val,
        y_val_scaled,
        X_test,
        y_test_scaled,
        lr=float(best_params["lr"]),
        weight_decay=float(best_params["weight_decay"]),
        num_epochs=int(refit_num_epochs),
        patience=int(patience),
        min_delta=float(min_delta),
        print_every=20,  # always show some output for the final run
        early_stop_on=scoring_norm,
    )

    test_metrics = eval_metrics(best_model, X_test, y_test_scaled, y_scaler = final_y_scaler)

    return {
        "best_estimator": {
            "model_state_dict": {k: v.detach().cpu() for k, v in best_model.state_dict().items()},
            "scaler": scaler,
            "y_scaler": final_y_scaler,
            "model_config": {
                "input_size": int(best_model.input_size),
                "hidden1": int(best_model.hidden1),
                "hidden2": int(best_model.hidden2),
                "dropout": float(best_model.dropout),
                "proj_dim": int(best_model.proj_dim),
                "lr": float(best_params["lr"]),
                "weight_decay": float(best_params["weight_decay"]),
            },
        },
        "best_params": best_params,
        "best_score": float(best_score),
        "cv_results": trial_results,
        "search_type": "ParameterSampler+KFold",
        "search_config": {
            "n_iter": int(n_iter),
            "cv_splits": int(cv_splits),
            "scoring": scoring,
            "scoring_normalized": scoring_norm,
            "seed": int(seed),
            "param_distributions": {k: list(v) for k, v in param_distributions.items()},
            "num_epochs": int(num_epochs),
            "refit_num_epochs": int(refit_num_epochs),
            "patience": int(patience),
            "min_delta": float(min_delta),
            "device": str(device_t),
        },
        "train_info": train_info,
        "test_metrics": test_metrics,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "seed": int(seed),
        "torch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "save_type": "torch",
    }

if __name__ == "__main__":
    # Example usage with fake data
    from sklearn.datasets import make_regression
    x, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    n_samples = len(y)
    train_idx = np.arange(int(0.8 * n_samples))
    test_idx = np.arange(int(0.8 * n_samples), n_samples)

    results = tune_mlp_with_fixed_split(
        x,
        y,
        train_idx,
        test_idx,
        seed=42,
        cv_splits=3,
        scoring="neg_mean_absolute_error",
        n_iter=10,
        num_epochs=100,
        patience=10,
        min_delta=1e-5,
        refit_num_epochs=200,
        device=None,
        print_every=0,
    )

    print("Best params:", results["best_params"])
    print("Best CV score:", results["best_score"])
    print("Test metrics:", results["test_metrics"])