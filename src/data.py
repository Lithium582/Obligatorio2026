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


def load_csv_with_fallback(local_path, remote_url, **read_csv_kwargs):
    """Carga un CSV local y utiliza una URL cuando el archivo no está disponible."""
    local_path = Path(local_path) if local_path is not None else None
    source = local_path if local_path is not None and local_path.is_file() else remote_url

    if source is None:
        raise FileNotFoundError(
            "No se encontró el archivo local y no se configuró una URL alternativa."
        )

    return load_csv(source, **read_csv_kwargs)


def combine_hotel_datasets(
    resort_df,
    city_df,
    hotel_column="HotelType",
):
    """Valida e integra los datasets H1 y H2 conservando el hotel de origen."""
    resort_columns = list(resort_df.columns)
    city_columns = list(city_df.columns)

    if resort_columns != city_columns:
        only_in_resort = sorted(set(resort_columns).difference(city_columns))
        only_in_city = sorted(set(city_columns).difference(resort_columns))
        raise ValueError(
            "Los datasets no comparten el mismo esquema y orden de columnas. "
            f"Sólo en Resort Hotel: {only_in_resort}. "
            f"Sólo en City Hotel: {only_in_city}."
        )

    resort_with_hotel = resort_df.copy()
    city_with_hotel = city_df.copy()
    resort_with_hotel.insert(0, hotel_column, "Resort Hotel")
    city_with_hotel.insert(0, hotel_column, "City Hotel")

    return pd.concat(
        [resort_with_hotel, city_with_hotel],
        ignore_index=True,
    )


def normalize_text_values(df, null_labels=("NULL",)):
    """Limpia espacios exteriores y convierte etiquetas nulas sin mutar el original."""
    normalized_df = df.copy()
    text_columns = normalized_df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    normalized_df[text_columns] = normalized_df[text_columns].apply(
        lambda column: column.str.strip()
    )
    normalized_df[text_columns] = normalized_df[text_columns].replace(
        list(null_labels),
        pd.NA,
    )

    return normalized_df


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
