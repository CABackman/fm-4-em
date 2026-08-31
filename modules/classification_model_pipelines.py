"""Utilities for tuning, training, and evaluating classification models."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


RANDOM_STATE = 5
CV_FOLDS = 5

DEFAULT_SVC_C_VALUES = [
    0.0001,
    0.001,
    0.01,
    0.05,
    0.1,
    0.5,
    1.0,
    10.0,
    50.0,
    100.0,
    500.0,
    1000.0,
]

DEFAULT_LOGREG_C_VALUES = [
    0.0001,
    0.001,
    0.01,
    0.05,
    0.1,
    0.5,
    1.0,
    10.0,
    100.0,
    1000.0,
]

RF_PARAM_GRID = {
    "max_features": ["log2", "sqrt"],
    "n_estimators": [5, 10, 25, 50, 100, 150, 200],
    "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
    "criterion": ["gini", "entropy"],
}

BAYES_VAR_SMOOTHING_VALUES = np.logspace(-12, -6, 13).tolist()

XGBOOST_PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 400, 600],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [3, 5, 7, 10],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}


def _get_lightgbm_classifier():
    """Import LightGBM only when a LightGBM model is requested."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as error:
        raise ImportError(
            "LightGBM was requested but is not installed in the active "
            "environment. Install the 'lightgbm' package first."
        ) from error
    return LGBMClassifier


def _get_hyperopt():
    """Import Hyperopt only when LightGBM tuning is requested."""
    try:
        from hyperopt import fmin, tpe
    except ImportError as error:
        raise ImportError(
            "LightGBM tuning requires Hyperopt. Install the 'hyperopt' "
            "package first."
        ) from error
    return fmin, tpe


def _get_xgboost_classifier():
    """Import XGBoost only when an XGBoost model is requested."""
    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise ImportError(
            "XGBoost was requested but is not installed in the active "
            "environment. Install the 'xgboost' package first."
        ) from error
    return XGBClassifier


def make_pipeline(classifier: Any) -> Pipeline:
    """Create a leakage-safe scaling and classification pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def _prefix_classifier_params(param_grid: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """Convert estimator parameter names to Pipeline parameter names."""
    return {
        f"classifier__{parameter}": values
        for parameter, values in param_grid.items()
    }


def _remove_classifier_prefix(params: dict[str, Any]) -> dict[str, Any]:
    """Return parameters in the format expected by ``run_model``."""
    return {
        parameter.removeprefix("classifier__"): value
        for parameter, value in params.items()
    }


def tune_svc(
    x_train,
    y_train,
    target: str,
    c_values: list[float] | None = None,
) -> dict[str, Any]:
    """Tune the regularisation strength of a scaled linear SVC."""
    pipeline = make_pipeline(LinearSVC(max_iter=100_000_000))
    param_grid = {"classifier__C": c_values or DEFAULT_SVC_C_VALUES}

    grid_search = GridSearchCV(pipeline, param_grid, cv=CV_FOLDS)
    grid_search.fit(x_train, y_train)

    print(f"Best score for {target} SVC: {grid_search.best_score_:.3f}")
    return _remove_classifier_prefix(grid_search.best_params_)


def tune_logreg(
    x_train,
    y_train,
    target: str,
    c_values: list[float] | None = None,
) -> dict[str, Any]:
    """Tune the regularisation strength of scaled logistic regression."""
    classifier = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=100_000,
    )
    pipeline = make_pipeline(classifier)
    param_grid = {"classifier__C": c_values or DEFAULT_LOGREG_C_VALUES}

    grid_search = GridSearchCV(pipeline, param_grid, cv=CV_FOLDS)
    grid_search.fit(x_train, y_train)

    print(
        f"Best score for {target} logistic regression: "
        f"{grid_search.best_score_:.3f}"
    )
    return _remove_classifier_prefix(grid_search.best_params_)


def tune_rf(
    x_train,
    y_train,
    target: str,
    param_grid: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """Tune a scaled random forest pipeline."""
    classifier = RandomForestClassifier(random_state=RANDOM_STATE)
    pipeline = make_pipeline(classifier)
    pipeline_param_grid = _prefix_classifier_params(param_grid or RF_PARAM_GRID)

    grid_search = GridSearchCV(pipeline, pipeline_param_grid, cv=CV_FOLDS)
    grid_search.fit(x_train, y_train)

    print(f"Best score for {target} random forest: {grid_search.best_score_:.3f}")
    return _remove_classifier_prefix(grid_search.best_params_)


def tune_bayes(
    x_train,
    y_train,
    target: str,
    var_smoothing_values: list[float] | None = None,
) -> dict[str, Any]:
    """Tune the variance smoothing of a scaled Gaussian Naive Bayes model."""
    pipeline = make_pipeline(GaussianNB())
    param_grid = {
        "classifier__var_smoothing": (
            var_smoothing_values or BAYES_VAR_SMOOTHING_VALUES
        )
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=CV_FOLDS)
    grid_search.fit(x_train, y_train)

    print(
        f"Best score for {target} Gaussian Naive Bayes: "
        f"{grid_search.best_score_:.3f}"
    )
    return _remove_classifier_prefix(grid_search.best_params_)


def tune_xgboost(
    x_train,
    y_train,
    target: str,
    param_distributions: dict[str, list[Any]] | None = None,
    n_iter: int = 20,
    n_jobs: int | None = None,
) -> dict[str, Any]:
    """Tune a scaled XGBoost classifier with randomized cross-validation."""
    XGBClassifier = _get_xgboost_classifier()
    classifier = XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,
        eval_metric="logloss",
    )
    pipeline = make_pipeline(classifier)
    pipeline_params = _prefix_classifier_params(
        param_distributions or XGBOOST_PARAM_DISTRIBUTIONS
    )
    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=pipeline_params,
        n_iter=n_iter,
        cv=cross_validation,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
    )
    random_search.fit(x_train, y_train)

    print(f"Best score for {target} XGBoost: {random_search.best_score_:.3f}")
    return _remove_classifier_prefix(random_search.best_params_)


def tune_lgbm(
    x_train,
    y_train,
    target: str,
    search_space: dict[str, Any],
    max_evals: int = 50,
) -> dict[str, Any]:
    """Tune a scaled LightGBM pipeline with Hyperopt."""
    LGBMClassifier = _get_lightgbm_classifier()
    fmin, tpe = _get_hyperopt()
    print(f"Tuning {target} LightGBM")
    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    def objective(params: dict[str, Any]) -> float:
        classifier_params = {
            "num_leaves": int(params["num_leaves"]),
            "colsample_bytree": float(params["colsample_bytree"]),
            "max_depth": int(params["max_depth"]),
        }
        classifier = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.01,
            random_state=RANDOM_STATE,
            verbose=-1,
            **classifier_params,
        )
        pipeline = make_pipeline(classifier)
        mean_score = cross_val_score(
            pipeline,
            x_train,
            y_train,
            cv=cross_validation,
        ).mean()

        # Hyperopt minimises the objective, so maximise accuracy via its negative.
        return -mean_score

    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
    )
    if "num_leaves" in best_params:
        best_params["num_leaves"] = int(best_params["num_leaves"])
    if "max_depth" in best_params:
        best_params["max_depth"] = int(best_params["max_depth"])
    if "colsample_bytree" in best_params:
        best_params["colsample_bytree"] = float(
            best_params["colsample_bytree"]
        )
    return best_params


def calculate_auc(x_test, y_test, model: Pipeline, target: str) -> None:
    """Print macro and weighted one-vs-rest ROC AUC scores."""
    if target == "binary":
        y_score = model.predict_proba(x_test)[:, 1]
    else:
        y_score = model.predict_proba(x_test)

    macro_auc = roc_auc_score(
        y_test,
        y_score,
        average="macro",
        multi_class="ovr",
    )
    weighted_auc = roc_auc_score(
        y_test,
        y_score,
        average="weighted",
        multi_class="ovr",
    )

    print(f"AUC-ROC (macro): {macro_auc:.3f}")
    print(f"AUC-ROC (weighted): {weighted_auc:.3f}")


def build_model(
    model_type: str,
    model_params: dict[str, Any],
) -> Pipeline:
    """Build a scaled pipeline for a supported classifier."""
    if model_type == "SVC":
        classifier = LinearSVC(max_iter=2_000_000_000, **model_params)
    elif model_type == "LogReg":
        classifier = LogisticRegression(
            max_iter=10_000_000,
            solver="newton-cg",
            **model_params,
        )
    elif model_type == "RandomForest":
        classifier = RandomForestClassifier(
            random_state=RANDOM_STATE,
            **model_params,
        )
    elif model_type in {"Bayes", "GaussianNB", "NaiveBayes"}:
        classifier = GaussianNB(**model_params)
    elif model_type == "XGBoost":
        XGBClassifier = _get_xgboost_classifier()
        classifier = XGBClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1,
            eval_metric="logloss",
            **model_params,
        )
    elif model_type == "LightGBM":
        LGBMClassifier = _get_lightgbm_classifier()
        classifier = LGBMClassifier(
            random_state=RANDOM_STATE,
            **model_params,
        )
    else:
        supported_models = (
            "SVC, LogReg, RandomForest, Bayes, XGBoost, LightGBM"
        )
        raise ValueError(
            f"Unknown model type {model_type!r}. Choose from: {supported_models}."
        )

    return make_pipeline(classifier)


def run_model(
    x_train,
    y_train,
    x_test,
    y_test,
    params: dict[str, Any],
    target: str,
    model_type: str,
) -> Pipeline:
    """Fit, evaluate, and return a scaled classification pipeline."""
    model_params = params if model_type == "LightGBM" else params[target]
    model = build_model(model_type, model_params)
    model.fit(x_train, y_train)

    print(
        f"{model_type} ({target}) training score: "
        f"{model.score(x_train, y_train):.3f}"
    )
    print(
        f"{model_type} ({target}) test score: "
        f"{model.score(x_test, y_test):.3f}"
    )

    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))

    matrix = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(matrix).plot(cmap="plasma")
    plt.show()

    # LinearSVC does not implement predict_proba.
    if model_type != "SVC":
        calculate_auc(x_test, y_test, model, target)

    return model
