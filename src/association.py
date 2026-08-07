import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.linear_model import LinearRegression


def prepare_categorical_for_association(series, minimum_rows=100):
    """Representa nulos y agrupa categorías infrecuentes para diagnóstico."""
    prepared = series.astype("string").fillna("Missing")
    category_counts = prepared.value_counts(dropna=False)
    rare_categories = category_counts.loc[category_counts.lt(minimum_rows)].index
    return prepared.mask(
        prepared.isin(rare_categories),
        f"Other (<{minimum_rows} rows)",
    )


def corrected_cramers_v(left, right):
    """Calcula V de Cramér corregida entre dos variables categóricas."""
    contingency = pd.crosstab(left, right)
    chi2, _, _, _ = chi2_contingency(contingency)
    observations = contingency.to_numpy().sum()
    rows, columns = contingency.shape
    phi_squared = chi2 / observations
    corrected_phi_squared = max(
        0,
        phi_squared - ((columns - 1) * (rows - 1)) / (observations - 1),
    )
    corrected_rows = rows - ((rows - 1) ** 2) / (observations - 1)
    corrected_columns = columns - ((columns - 1) ** 2) / (observations - 1)
    denominator = min(corrected_columns - 1, corrected_rows - 1)
    return (
        np.sqrt(corrected_phi_squared / denominator)
        if denominator > 0
        else 0.0
    )


def correlation_ratio(categories, numeric_values):
    """Calcula eta entre una variable categórica y una numérica."""
    analysis_frame = pd.DataFrame(
        {"category": categories, "value": numeric_values}
    ).dropna(subset=["value"])
    if analysis_frame.empty or analysis_frame["value"].var(ddof=0) == 0:
        return 0.0

    overall_mean = analysis_frame["value"].mean()
    grouped = analysis_frame.groupby("category", observed=True)["value"].agg(
        ["count", "mean"]
    )
    between_group_variation = (
        grouped["count"] * (grouped["mean"] - overall_mean) ** 2
    ).sum()
    total_variation = ((analysis_frame["value"] - overall_mean) ** 2).sum()
    return (
        np.sqrt(between_group_variation / total_variation)
        if total_variation > 0
        else 0.0
    )


def build_focused_association_table(
    X,
    focus_features,
    categorical_features,
    minimum_category_rows=100,
):
    """Compara features focales con el resto usando métricas por tipo."""
    focus_features = list(focus_features)
    categorical_features = set(categorical_features)
    missing = sorted(set(focus_features).difference(X.columns))
    if missing:
        raise ValueError(f"Faltan features focales: {missing}")

    prepared_categories = {
        feature: prepare_categorical_for_association(
            X[feature], minimum_rows=minimum_category_rows
        )
        for feature in categorical_features
        if feature in X.columns
    }
    records = []
    seen_pairs = set()

    for focus_feature in focus_features:
        for other_feature in X.columns:
            if focus_feature == other_feature:
                continue
            pair_key = frozenset((focus_feature, other_feature))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            focus_is_categorical = focus_feature in categorical_features
            other_is_categorical = other_feature in categorical_features

            if focus_is_categorical and other_is_categorical:
                association = corrected_cramers_v(
                    prepared_categories[focus_feature],
                    prepared_categories[other_feature],
                )
                metric = "Corrected Cramer's V"
            elif focus_is_categorical or other_is_categorical:
                categorical_feature = (
                    focus_feature if focus_is_categorical else other_feature
                )
                numeric_feature = (
                    other_feature if focus_is_categorical else focus_feature
                )
                association = correlation_ratio(
                    prepared_categories[categorical_feature],
                    X[numeric_feature],
                )
                metric = "Correlation ratio eta"
            else:
                association = abs(
                    X[focus_feature].corr(X[other_feature], method="spearman")
                )
                metric = "Absolute Spearman"

            records.append(
                {
                    "engineered_feature": focus_feature,
                    "compared_feature": other_feature,
                    "metric": metric,
                    "association": association,
                }
            )

    return pd.DataFrame(records).sort_values(
        "association", ascending=False
    ).reset_index(drop=True)


def calculate_numeric_vif(X, numeric_features):
    """Calcula VIF mediante regresiones auxiliares sobre variables numéricas."""
    numeric_data = X[list(numeric_features)].apply(pd.to_numeric, errors="coerce")
    numeric_data = numeric_data.fillna(numeric_data.median())
    usable_features = [
        feature for feature in numeric_data.columns
        if numeric_data[feature].nunique() > 1
    ]
    records = []

    for feature in usable_features:
        other_features = [item for item in usable_features if item != feature]
        auxiliary_model = LinearRegression(n_jobs=-1)
        auxiliary_model.fit(numeric_data[other_features], numeric_data[feature])
        r_squared = auxiliary_model.score(
            numeric_data[other_features], numeric_data[feature]
        )
        vif = np.inf if r_squared >= 1 - 1e-12 else 1 / (1 - r_squared)
        records.append(
            {
                "feature": feature,
                "r_squared_against_other_numeric_features": r_squared,
                "vif": vif,
            }
        )

    return pd.DataFrame(records).sort_values("vif", ascending=False).reset_index(
        drop=True
    )
