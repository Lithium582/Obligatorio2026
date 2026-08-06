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

CLEANED_BASE_COLUMNS_TO_EXCLUDE = (
    "ArrivalDateWeekNumber",
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


def remove_redundant_base_features(
    X,
    columns_to_exclude=CLEANED_BASE_COLUMNS_TO_EXCLUDE,
):
    """Elimina redundancias acordadas después de establecer el baseline."""
    columns_to_exclude = list(columns_to_exclude)
    missing_columns = sorted(set(columns_to_exclude).difference(X.columns))
    if missing_columns:
        raise ValueError(
            f"No se pueden excluir features inexistentes: {missing_columns}"
        )
    return X.drop(columns=columns_to_exclude).copy()


def make_exact_feature_groups(X):
    """Genera un identificador estable por combinación exacta de predictores."""
    return pd.util.hash_pandas_object(X, index=False).astype("uint64")


def add_stay_composition_features(X):
    """Agrega totales interpretables de huéspedes y noches."""
    engineered = X.copy()
    engineered["TotalGuests"] = engineered[
        ["Adults", "Children", "Babies"]
    ].sum(axis=1, min_count=1)
    engineered["TotalNights"] = engineered[
        ["StaysInWeekendNights", "StaysInWeekNights"]
    ].sum(axis=1, min_count=1)
    return engineered


def add_profile_indicator_features(X):
    """Agrega indicadores de origen, espera e historial del cliente."""
    engineered = X.copy()
    engineered["IsDomestic"] = (
        engineered["Country"]
        .eq("PRT")
        .where(engineered["Country"].notna())
        .astype("float64")
    )
    engineered["WasOnWaitingList"] = (
        engineered["DaysInWaitingList"].gt(0).astype("int8")
    )
    engineered["HasPreviousCancellation"] = (
        engineered["PreviousCancellations"].gt(0).astype("int8")
    )

    previous_bookings = (
        engineered["PreviousCancellations"]
        + engineered["PreviousBookingsNotCanceled"]
    )
    engineered["PreviousCancellationRate"] = (
        engineered["PreviousCancellations"]
        .div(previous_bookings.where(previous_bookings.gt(0)))
        .fillna(0.0)
    )
    return engineered


def add_waiting_list_interaction_features(X):
    """Permite un efecto diferente de la lista de espera en cada hotel."""
    engineered = X.copy()
    if "WasOnWaitingList" not in engineered.columns:
        engineered["WasOnWaitingList"] = (
            engineered["DaysInWaitingList"].gt(0).astype("int8")
        )
    engineered["CityHotelWasOnWaitingList"] = (
        engineered["HotelType"].eq("City Hotel")
        & engineered["WasOnWaitingList"].eq(1)
    ).astype("int8")
    return engineered


def build_first_engineered_feature_set(X):
    """Construye el primer lote acordado sin eliminar variables originales."""
    engineered = add_stay_composition_features(X)
    engineered = add_profile_indicator_features(engineered)
    engineered = add_waiting_list_interaction_features(engineered)
    return engineered
