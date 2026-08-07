import numpy as np
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

MONTH_TO_NUMBER = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

CLEANED_BASE_COLUMNS_TO_EXCLUDE = (
    "ArrivalDateWeekNumber",
)

TREE_PRUNING_COLUMNS_TO_EXCLUDE = (
    "Babies",
    "Children",
    "IsRepeatedGuest",
    "HasCompany",
    "ReservedRoomType",
    "ArrivalDateDayOfMonth",
    "Meal",
)

LOGISTIC_REGRESSION_FIRST_BATCH_EXCLUSIONS = (
    "ArrivalDateWeekNumber",
    "TotalGuests",
    "TotalNights",
    "PreviousCancellations",
    "DaysInWaitingList",
    "IsDomestic",
    "HasPreviousCancellation",
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


def get_feature_types(
    X,
    categorical_features=BASE_CATEGORICAL_FEATURES,
    require_all_categorical=True,
):
    """Valida y separa las columnas categóricas y numéricas del conjunto base."""
    categorical_features = list(categorical_features)
    missing_columns = sorted(set(categorical_features).difference(X.columns))
    if missing_columns and require_all_categorical:
        raise ValueError(f"Faltan variables categóricas esperadas: {missing_columns}")

    inferred_categorical_features = list(
        X.select_dtypes(include=["object", "string", "category"]).columns
    )
    categorical_features = list(dict.fromkeys([
        *categorical_features,
        *inferred_categorical_features,
    ]))
    categorical_features = [
        feature for feature in categorical_features if feature in X.columns
    ]

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


def build_tree_pruned_feature_set(X):
    """Construye el conjunto de 17 features elegido mediante el árbol."""
    cleaned = remove_redundant_base_features(X)
    missing_columns = sorted(
        set(TREE_PRUNING_COLUMNS_TO_EXCLUDE).difference(cleaned.columns)
    )
    if missing_columns:
        raise ValueError(
            "No se pueden aplicar las exclusiones del árbol: "
            f"{missing_columns}"
        )
    return cleaned.drop(columns=list(TREE_PRUNING_COLUMNS_TO_EXCLUDE)).copy()


def build_tree_model_feature_sets(X):
    """Devuelve las bases comparables para Random Forest y XGBoost."""
    return {
        "Cleaned base | 24 features": remove_redundant_base_features(X),
        "Tree-pruned | 17 features": build_tree_pruned_feature_set(X),
    }


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
    engineered = add_domestic_indicator(X)
    engineered = add_waiting_list_indicator(engineered)
    engineered = add_previous_cancellation_indicator(engineered)
    engineered = add_previous_cancellation_rate(engineered)
    return engineered


def add_domestic_indicator(X):
    """Agrega un indicador de país portugués conservando los nulos."""
    engineered = X.copy()
    engineered["IsDomestic"] = (
        engineered["Country"]
        .eq("PRT")
        .where(engineered["Country"].notna())
        .astype("float64")
    )
    return engineered


def add_waiting_list_indicator(X):
    """Agrega un indicador de tiempo positivo en lista de espera."""
    engineered = X.copy()
    engineered["WasOnWaitingList"] = (
        engineered["DaysInWaitingList"].gt(0).astype("int8")
    )
    return engineered


def add_previous_cancellation_indicator(X):
    """Agrega un indicador de al menos una cancelación anterior."""
    engineered = X.copy()
    engineered["HasPreviousCancellation"] = (
        engineered["PreviousCancellations"].gt(0).astype("int8")
    )
    return engineered


def add_previous_cancellation_rate(X):
    """Agrega la proporción histórica de cancelaciones conocidas."""
    engineered = X.copy()

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


def build_logistic_first_batch_candidate_set(X):
    """Construye el candidato lineal parsimonioso elegido por ablación."""
    cleaned = remove_redundant_base_features(X)
    engineered = add_previous_cancellation_rate(cleaned)
    engineered = engineered.drop(columns=["PreviousCancellations"])
    engineered = add_waiting_list_indicator(engineered)
    engineered = add_waiting_list_interaction_features(engineered)
    engineered = engineered.drop(columns=["DaysInWaitingList"])

    unexpected_features = sorted(
        set(LOGISTIC_REGRESSION_FIRST_BATCH_EXCLUSIONS).intersection(
            engineered.columns
        )
    )
    if unexpected_features:
        raise ValueError(
            "El candidato lineal conserva features excluidas: "
            f"{unexpected_features}"
        )
    return engineered


def build_logistic_redundant_first_batch_set(X):
    """Construye el candidato de máxima media conservando redundancias."""
    cleaned = remove_redundant_base_features(X)
    engineered = add_profile_indicator_features(cleaned)
    engineered = add_waiting_list_interaction_features(engineered)
    return engineered


def add_cyclic_month_features(X, drop_original=False):
    """Representa el mes mediante seno y coseno."""
    engineered = X.copy()
    month_number = engineered["ArrivalDateMonth"].map(MONTH_TO_NUMBER)
    if month_number.isna().any():
        unknown_months = sorted(
            engineered.loc[month_number.isna(), "ArrivalDateMonth"]
            .dropna()
            .unique()
        )
        raise ValueError(f"Meses desconocidos: {unknown_months}")

    month_angle = 2 * np.pi * (month_number - 1) / 12
    engineered["ArrivalMonthSin"] = np.sin(month_angle)
    engineered["ArrivalMonthCos"] = np.cos(month_angle)
    if drop_original:
        engineered = engineered.drop(columns=["ArrivalDateMonth"])
    return engineered


def add_cyclic_day_of_week_features(X):
    """Deriva la fecha de llegada y representa el día semanal cíclicamente."""
    engineered = X.copy()
    month_number = engineered["ArrivalDateMonth"].map(MONTH_TO_NUMBER)
    arrival_date = pd.to_datetime(
        {
            "year": engineered["ArrivalDateYear"],
            "month": month_number,
            "day": engineered["ArrivalDateDayOfMonth"],
        }
    )
    weekday_angle = 2 * np.pi * arrival_date.dt.dayofweek / 7
    engineered["ArrivalDayOfWeekSin"] = np.sin(weekday_angle)
    engineered["ArrivalDayOfWeekCos"] = np.cos(weekday_angle)
    return engineered


def add_log_lead_time_feature(X, drop_original=False):
    """Agrega log1p de LeadTime para representar rendimientos decrecientes."""
    engineered = X.copy()
    if engineered["LeadTime"].lt(0).any():
        raise ValueError("LeadTime contiene valores negativos.")
    engineered["LogLeadTime"] = np.log1p(engineered["LeadTime"])
    if drop_original:
        engineered = engineered.drop(columns=["LeadTime"])
    return engineered


def add_lead_time_band_feature(X, drop_original=False):
    """Agrupa LeadTime en intervalos operativos definidos en Feature Analysis."""
    engineered = X.copy()
    engineered["LeadTimeBand"] = pd.cut(
        engineered["LeadTime"],
        bins=[-1, 0, 7, 30, 90, 180, 365, np.inf],
        labels=[
            "Same day",
            "1-7 days",
            "8-30 days",
            "31-90 days",
            "91-180 days",
            "181-365 days",
            "366+ days",
        ],
    )
    if drop_original:
        engineered = engineered.drop(columns=["LeadTime"])
    return engineered


def add_special_request_indicator(X, drop_original=False):
    """Agrega un indicador de al menos una solicitud especial."""
    engineered = X.copy()
    engineered["HasSpecialRequests"] = (
        engineered["TotalOfSpecialRequests"].gt(0).astype("int8")
    )
    if drop_original:
        engineered = engineered.drop(columns=["TotalOfSpecialRequests"])
    return engineered


def build_logistic_second_batch_candidate_set(X):
    """Construye el conjunto lineal elegido tras el segundo lote."""
    engineered = build_logistic_first_batch_candidate_set(X)
    engineered = add_lead_time_band_feature(engineered, drop_original=True)
    engineered = add_special_request_indicator(engineered)
    return engineered


def add_parking_indicator(X, drop_original=False):
    """Agrega un indicador de al menos un espacio de estacionamiento."""
    engineered = X.copy()
    engineered["HasParking"] = (
        engineered["RequiredCarParkingSpaces"].gt(0).astype("int8")
    )
    if drop_original:
        engineered = engineered.drop(columns=["RequiredCarParkingSpaces"])
    return engineered


def build_second_batch_diagnostic_set(X):
    """Agrega todas las representaciones del segundo lote para diagnóstico."""
    engineered = add_cyclic_month_features(X)
    engineered = add_cyclic_day_of_week_features(engineered)
    engineered = add_log_lead_time_feature(engineered)
    engineered = add_lead_time_band_feature(engineered)
    engineered = add_special_request_indicator(engineered)
    engineered = add_parking_indicator(engineered)
    return engineered
