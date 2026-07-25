from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_tabular_preprocessor(
    numeric_features,
    categorical_features,
    scale_numeric=False,
    numeric_imputation="median",
    categorical_imputation="most_frequent",
    drop_categorical=None,
):
    """Construye un preprocesador tabular configurable y libre de leakage."""
    numeric_steps = [
        ("imputer", SimpleImputer(strategy=numeric_imputation)),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=categorical_imputation)),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop=drop_categorical,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, list(numeric_features)),
            (
                "categorical",
                categorical_transformer,
                list(categorical_features),
            ),
        ],
        remainder="drop",
    )
