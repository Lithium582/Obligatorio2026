import json
from pathlib import Path

import pandas as pd


def create_experiment_result(
    experiment,
    model,
    feature_set,
    cv_mean,
    cv_std,
    validation_score=None,
    best_params=None,
    notes=None,
):
    """Crea un registro comparable para una iteración experimental."""
    return {
        "experiment": experiment,
        "model": model,
        "feature_set": feature_set,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "validation_score": validation_score,
        "best_params": (
            json.dumps(best_params, ensure_ascii=False)
            if best_params is not None
            else None
        ),
        "notes": notes,
    }


def append_result(results, result):
    """Agrega un resultado sin modificar el DataFrame recibido."""
    row = pd.DataFrame([result])
    if results is None or results.empty:
        return row
    return pd.concat([results, row], ignore_index=True)


def save_results(results, filepath):
    """Guarda la tabla de experimentos y crea su directorio si hace falta."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(filepath, index=False)


def load_results(filepath):
    """Carga resultados anteriores o devuelve una tabla vacía."""
    filepath = Path(filepath)
    if not filepath.exists():
        return pd.DataFrame()
    return pd.read_csv(filepath)
