from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sklearn

def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression metrics.

    Params:
    y_true : np.ndarray
        Ground-truth targets of shape (n_samples,) or (n_samples, 1).
    y_pred : np.ndarray
        Model predictions of shape (n_samples,) or (n_samples, 1).

    Returns
    -------
    dict
        Dictionary containing:
            MAE
            RMSE
            R2
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": rmse, "R2": float(r2)}


def fit_linear_model_with_fixed_split(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    seed: int = 0,
    cv_splits: int = 5,
    alphas: np.ndarray = np.logspace(-2, 1, 60),
    l1_ratios: np.ndarray = (0.0, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.035),
    max_iter: int = 100000,
    scoring: str = "neg_mean_absolute_error", ) -> pd.DataFrame:
    """
    Train and evaluate linear probe baselines  on frozen embeddings.

    This function:
        Splits data into TRAIN / TEST using provided indices.
        Hyperparameter tuning is performed using K-fold CV on TRAIN only.

    Pipeline:
        StandardScaler -> LinearModel

    so that feature scaling is learned on TRAIN folds only during CV.

    Params:
    ---
    X : np.ndarray
        Embeddings with shape (n_samples, n_features).
    y : np.ndarray
        Target array of shape (n_samples,) or (n_samples, 1).
    train_idx, test_idx : np.ndarray
        Index arrays defining the fixed TRAIN / TEST splits.
        Must have the same row ordering as in each embedding matrix.

    seed : int, default=0
        Random seed controlling KFold shuffling.
    cv_splits : int, default=5
        
    alphas : np.ndarray, default = np.logspace(-2, 6, 60)
        Regularization strengths to grid search.

    l1_ratios : tuple, default = (0.0, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.035)
        ElasticNet mixing parameter candidates. 0 = pure Ridge, 1 = pure Lasso 
        (the models are all basically 95-100% Ridge)
    max_iter : int, default=50_000
        Maximum iterations
    scoring : str, default="neg_mean_absolute_error"
        GridSearchCV scoring string used for hyperparameter selection on TRAIN folds.

    Returns
    -------
    dict : ckpt info
    """
    y = np.asarray(y).ravel()

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    X = np.asarray(X)
    if X.shape[0] != len(y):
        raise ValueError(f" X has {X.shape[0]} rows but y has {len(y)}")

    # Fixed split
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test   = X[test_idx],  y[test_idx]

    # ElasticNet
    enet_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(max_iter=max_iter, random_state=seed)),
    ])
    enet_gs = RandomizedSearchCV(
        enet_pipe,
        param_distributions={
            "model__alpha": alphas,
            "model__l1_ratio": l1_ratios,
        },
        n_iter=50,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        random_state=seed
    )
    enet_gs.fit(X_train, y_train)
    enet_best = enet_gs.best_estimator_

    pred_train = enet_best.predict(X_train)
    pred_test = enet_best.predict(X_test)

    m_train = _regression_metrics(y_train, pred_train)
    m_test = _regression_metrics(y_test, pred_test)
    
    print("Done!")
    search = enet_gs
    search_type = type(enet_gs).__name__
    out = {
        "best_estimator": getattr(search, "best_estimator_", None),
        "best_params": getattr(search, "best_params_", None),
        "best_score": getattr(search, "best_score_", None),
        "cv_results": getattr(search, "cv_results_", None),
        "search_type": search_type,
        "search_config": {
            "scoring": scoring,
            "cv": str(cv),
            "refit": getattr(search, "refit", None),
            "n_jobs": getattr(search, "n_jobs", None),
            "verbose": getattr(search, "verbose", None),
            "error_score": getattr(search, "error_score", None),
            "return_train_score": getattr(search, "return_train_score", None),
        },
        "train_metrics": m_train,
        "test_metrics": m_test,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "seed": seed,
        "sklearn_version": sklearn.__version__,
        "save_type": "sklearn",
    }

    # Add method-specific config safely
    if hasattr(search, "param_grid"):  # GridSearchCV
        out["search_config"]["param_grid"] = search.param_grid
    if hasattr(search, "param_distributions"):  # RandomizedSearchCV
        out["search_config"]["param_distributions"] = search.param_distributions
        out["search_config"]["n_iter"] = getattr(search, "n_iter", None)
        out["search_config"]["random_state"] = getattr(search, "random_state", None)

    return out