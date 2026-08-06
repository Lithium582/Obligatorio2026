import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
)

from src.config import RANDOM_STATE


def get_cross_validation(
    task_type="classification",
    n_splits=5,
    random_state=RANDOM_STATE,
):
    """Devuelve folds reproducibles para clasificación o regresión."""
    common = {
        "n_splits": n_splits,
        "shuffle": True,
        "random_state": random_state,
    }
    if task_type == "classification":
        return StratifiedKFold(**common)
    if task_type == "regression":
        return KFold(**common)
    raise ValueError("task_type debe ser 'classification' o 'regression'.")


def get_stratified_group_cross_validation(
    n_splits=5,
    random_state=RANDOM_STATE,
):
    """Crea folds estratificados sin separar observaciones del mismo grupo."""
    return StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def cross_validate_model(
    estimator,
    X,
    y,
    scoring,
    task_type="classification",
    n_splits=5,
    n_jobs=-1,
    cv=None,
    groups=None,
):
    """Evalúa un estimador con una estrategia común de validación."""
    if cv is None:
        cv = get_cross_validation(
            task_type=task_type,
            n_splits=n_splits,
        )
    return cross_val_score(
        estimator,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        groups=groups,
    )


def summarize_cv_scores(scores):
    """Resume una colección de scores de validación cruzada."""
    scores = np.asarray(scores)
    return {
        "cv_mean": scores.mean(),
        "cv_std": scores.std(),
        "cv_scores": scores,
    }


def run_grid_search(
    estimator,
    param_grid,
    X,
    y,
    scoring,
    task_type="classification",
    n_splits=5,
    n_jobs=-1,
    refit=True,
    return_train_score=True,
    cv=None,
    groups=None,
):
    """Ejecuta una búsqueda usando los mismos folds que el resto del proyecto."""
    if cv is None:
        cv = get_cross_validation(
            task_type=task_type,
            n_splits=n_splits,
        )
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=refit,
        return_train_score=return_train_score,
    )
    search.fit(X, y, groups=groups)
    return search


def get_grid_results(search):
    """Devuelve los resultados ordenados por ranking de validación."""
    return (
        pd.DataFrame(search.cv_results_)
        .sort_values("rank_test_score")
        .reset_index(drop=True)
    )


def evaluate_binary_classifier(model, X, y, threshold=0.5):
    """Evalúa probabilidades y clases de un clasificador binario."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("El modelo debe implementar predict_proba.")

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, probabilities),
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "threshold": threshold,
    }


def evaluate_candidates_on_holdout(
    search,
    base_estimator,
    X_train,
    y_train,
    X_validation,
    y_validation,
    top_n=5,
    threshold=0.5,
):
    """Compara los mejores candidatos de CV sobre un holdout separado."""
    candidates = get_grid_results(search).head(top_n)
    rows = []

    for _, candidate in candidates.iterrows():
        estimator = clone(base_estimator).set_params(**candidate["params"])
        estimator.fit(X_train, y_train)
        metrics = evaluate_binary_classifier(
            estimator,
            X_validation,
            y_validation,
            threshold=threshold,
        )
        rows.append(
            {
                "params": candidate["params"],
                "cv_score_mean": candidate["mean_test_score"],
                "cv_score_std": candidate["std_test_score"],
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
        )

    return pd.DataFrame(rows)
