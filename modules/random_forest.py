from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Optional, Dict, Any

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    return {"MAE": float(mae), "RMSE": rmse, "R2": float(r2)}


def fit_random_forest_with_fixed_split(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    model_type: str = "random_forest",
    seed: int = 42,
    cv_splits: int = 3,
    scoring: str = "neg_mean_absolute_error",
    n_iter: int = 20,
    param_distributions: Optional[Dict[str, Any]] = None,
    **param_overrides,
) -> dict:
    """
    Train and evaluate model with CV over train set.

    Hyperparameter search space can be passed via:
    - param_distributions (full dict)
    - individual keyword overrides (e.g. model__max_depth=[10,20])

    model_type: "random_forest" (default) or "xgboost"
    """

    y = np.asarray(y).ravel()
    X = np.asarray(X)

    if X.shape[0] != len(y):
        raise ValueError(f"X has {X.shape[0]} rows but y has {len(y)}")

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    # Fixed split
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model_type_norm = str(model_type).lower().strip()

    if model_type_norm in ("random_forest", "rf", "randomforest"):
        model = RandomForestRegressor(
            random_state=seed,
            n_jobs=1,
        )
        rf_pipe = Pipeline([
            ("model", model),
        ])

        # Default search space
        default_param_space = {
            "model__n_estimators": [300, 400, 500],
            "model__max_depth": [8, 12, 16, 20, None],
            "model__min_samples_leaf": [2, 5, 10, 20, 40],
            "model__min_samples_split": [2, 5, 10],
            "model__max_features": ["sqrt", "log2", 0.2, 0.3, 0.5],
        }

        print("Randomized Search...")

    elif model_type_norm in ("xgboost", "xgb"):
        if XGBRegressor is None:
            raise ImportError(
                "model_type='xgboost' requires xgboost to be installed. "
                "Install with: pip install xgboost"
    )

        model = XGBRegressor(
            random_state=seed,
            n_estimators=500,
            objective="reg:squarederror",
            n_jobs=1,
            verbosity=0,
        )
        rf_pipe = Pipeline([
            ("model", model),
        ])

        # Default search space (approx. similar tuning budget/runtime to RF)
        default_param_space = {
            "model__n_estimators": [300, 400, 500, 700],
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.5, 0.7, 0.85, 1.0],
            "model__min_child_weight": [1, 2, 5, 10],
            "model__reg_alpha": [0.0, 1e-4, 1e-3, 1e-2, 0.1],
            "model__reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
        }

        print("Randomized Search...")

    else:
        raise ValueError(
            f"Unsupported model_type={model_type!r}. "
            "Expected 'random_forest' (default) or 'xgboost'."
        )

    # If full param_distributions provided, use it
    if param_distributions is not None:
        search_space = param_distributions.copy()
    else:
        search_space = default_param_space.copy()

    # Apply individual overrides
    search_space.update(param_overrides)

    rf_gs = RandomizedSearchCV(
        rf_pipe,
        param_distributions=search_space,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        random_state=seed,
        verbose=100,
    )

    rf_gs.fit(X_train, y_train)

    # Evaluate
    rf_best = rf_gs.best_estimator_
    pred_train = rf_best.predict(X_train)
    pred_test = rf_best.predict(X_test)

    m_train = _regression_metrics(y_train, pred_train)
    m_test = _regression_metrics(y_test, pred_test)

    print("Done!")
    return {
        "best_estimator": rf_gs.best_estimator_,   
        "best_params": rf_gs.best_params_,
        "best_score": rf_gs.best_score_,
        "cv_results": rf_gs.cv_results_,           
        "search_type": "RandomizedSearchCV",
        "search_config": {
            "n_iter": rf_gs.n_iter,
            "scoring": scoring,
            "cv": str(cv),
            "random_state": rf_gs.random_state,
        },
        "train_metrics": m_train,
        "test_metrics": m_test,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "seed": seed,
        "sklearn_version": sklearn.__version__,
        "save_type": "sklearn",
    }

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    train_idx = np.arange(800)
    test_idx = np.arange(800, 1000)

    results_rf = fit_random_forest_with_fixed_split(
        X, y, train_idx, test_idx, model_type="random_forest", seed=42
    )
    print("Random Forest Results:", results_rf)

    if XGBRegressor is not None:
        results_xgb = fit_random_forest_with_fixed_split(
            X, y, train_idx, test_idx, model_type="xgboost", seed=42
        )
        print("XGBoost Results:", results_xgb)