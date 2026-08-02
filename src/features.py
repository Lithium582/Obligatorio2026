import pandas as pd

from src.config import TARGET_COL
from src.data import split_features_target


BASE_COLUMNS_TO_EXCLUDE = (
    "ReservationStatus",
    "ReservationStatusDate",
    "AssignedRoomType",
    "BookingChanges",
    "DepositType",
    "ADR",
    "Agent",
    "Company",
)

BASE_CATEGORICAL_FEATURES = (
    "HotelType",
    "ArrivalDateMonth",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "ReservedRoomType",
    "CustomerType",
)


def build_base_feature_set(df, target_col=TARGET_COL):
    """Construye el conjunto original depurado, sin features de ingeniería."""
    required_columns = {
        target_col,
        "Agent",
        "Company",
        *BASE_COLUMNS_TO_EXCLUDE,
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(
            f"Faltan columnas requeridas para el conjunto base: {missing_columns}"
        )

    prepared_df = df.copy()
    prepared_df["HasAgent"] = prepared_df["Agent"].notna().astype("int8")
    prepared_df["HasCompany"] = prepared_df["Company"].notna().astype("int8")

    return split_features_target(
        prepared_df,
        target_col=target_col,
        columns_to_drop=BASE_COLUMNS_TO_EXCLUDE,
    )


def get_feature_types(X, categorical_features=BASE_CATEGORICAL_FEATURES):
    """Valida y separa las columnas categóricas y numéricas del conjunto base."""
    categorical_features = list(categorical_features)
    missing_columns = sorted(set(categorical_features).difference(X.columns))
    if missing_columns:
        raise ValueError(f"Faltan variables categóricas esperadas: {missing_columns}")

    numeric_features = [
        column for column in X.columns if column not in categorical_features
    ]
    return numeric_features, categorical_features


def make_exact_feature_groups(X):
    """Genera un identificador estable por combinación exacta de predictores."""
    return pd.util.hash_pandas_object(X, index=False).astype("uint64")
