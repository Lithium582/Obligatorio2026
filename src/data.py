from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE


def load_csv(filepath, **read_csv_kwargs):
    """Carga un CSV desde una ruta local o una URL."""
    return pd.read_csv(filepath, **read_csv_kwargs)


def load_raw_csv(filename, raw_data_dir):
    """Carga un CSV del directorio de datos originales."""
    return load_csv(Path(raw_data_dir) / filename)


def split_features_target(df, target_col, columns_to_drop=None):
    """Separa predictores y objetivo sin modificar el DataFrame original."""
    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no existe.")

    columns_to_drop = list(columns_to_drop or [])
    unknown_columns = sorted(set(columns_to_drop).difference(df.columns))
    if unknown_columns:
        raise ValueError(
            f"No se pueden eliminar columnas inexistentes: {unknown_columns}"
        )

    X = df.drop(columns=[target_col, *columns_to_drop])
    y = df[target_col].copy()
    return X, y


def make_train_validation_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=True,
):
    """Crea un holdout reproducible; permite estratificar en clasificación."""
    stratify_values = y if stratify else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )
