"""Feature engineering y encoding del modelo de crímenes de Chicago.

Reproduce el pipeline del notebook `Crimenes_en_Chicago_Federico_Romero_Ailen_Muñoz.ipynb`
(TP de Aprendizaje de Máquina I): 3 features temporales derivadas de `Date`
(codificadas cíclicamente), Binary Encoding para `Location Description`/`Beat`
(alta cardinalidad) y Ordinal Encoding para los booleanos `Arrest`/`Domestic`.

Este módulo es puro (sin I/O a S3/MLflow) para poder empaquetarlo junto al
modelo servido (ver `amq2/pipeline.py`) y reutilizarlo tanto en el DAG de
entrenamiento (sobre un DataFrame de ~190k filas) como en la API de FastAPI
(sobre un DataFrame de 1 fila).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from category_encoders import BinaryEncoder
from sklearn.preprocessing import OrdinalEncoder

TARGET_COLUMN = "Primary Type"

# Features seleccionadas en el notebook (Community Area quedó excluida por ser
# redundante con Beat).
RAW_FEATURE_COLUMNS = ["Location Description", "Beat", "Arrest", "Domestic", "Date"]
SELECTED_FEATURES = [
    "Location Description",
    "Beat",
    "Arrest",
    "Domestic",
    "Day of week",
    "Month",
    "Moment of day",
]

BINARY_ENCODED_COLUMNS = ["Location Description", "Beat"]
ORDINAL_ENCODED_COLUMNS = ["Arrest", "Domestic"]

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
MOMENT_ORDER = ["MORNING", "AFTERNOON", "NIGHT"]

_DAY_TO_NUM = {d: i for i, d in enumerate(DAY_ORDER)}
_MONTH_TO_NUM = {m: i for i, m in enumerate(MONTH_ORDER)}
_MOMENT_TO_NUM = {m: i for i, m in enumerate(MOMENT_ORDER)}


def get_moment_of_day(hour: int) -> str:
    """Franja horaria (3 tramos de 8hs) a la que pertenece `hour` (0-23)."""
    if 6 <= hour < 14:
        return "MORNING"
    if 14 <= hour < 22:
        return "AFTERNOON"
    return "NIGHT"


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva `Day of week`, `Month` y `Moment of day` desde `Date` y la elimina."""
    df = df.copy()
    date = pd.to_datetime(df["Date"])
    df["Day of week"] = date.dt.day_name()
    df["Month"] = date.dt.month_name()
    df["Moment of day"] = date.dt.hour.apply(get_moment_of_day)
    return df.drop(columns=["Date"])


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """`Location Description` faltante es un valor estructural -> 'VIRTUAL'."""
    df = df.copy()
    df["Location Description"] = df["Location Description"].fillna("VIRTUAL")
    return df


def build_binary_encoder() -> BinaryEncoder:
    """Instancia (sin ajustar) el Binary Encoder para `Location Description`/`Beat`."""
    return BinaryEncoder(cols=BINARY_ENCODED_COLUMNS, return_df=True)


def build_ordinal_encoder() -> OrdinalEncoder:
    """Instancia (sin ajustar) el Ordinal Encoder para los booleanos `Arrest`/`Domestic`."""
    return OrdinalEncoder(
        categories=[[False, True], [False, True]],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )


def _add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Proyecta `Day of week`/`Month`/`Moment of day` sobre un círculo (seno/coseno)."""
    out = pd.DataFrame(index=df.index)
    dow = df["Day of week"].map(_DAY_TO_NUM)
    month = df["Month"].map(_MONTH_TO_NUM)
    moment = df["Moment of day"].map(_MOMENT_TO_NUM)
    out["Day of week_sin"] = np.sin(2 * np.pi * dow / 7)
    out["Day of week_cos"] = np.cos(2 * np.pi * dow / 7)
    out["Month_sin"] = np.sin(2 * np.pi * month / 12)
    out["Month_cos"] = np.cos(2 * np.pi * month / 12)
    out["Moment of day_sin"] = np.sin(2 * np.pi * moment / 3)
    out["Moment of day_cos"] = np.cos(2 * np.pi * moment / 3)
    return out


def encode_features(
    df: pd.DataFrame, binary_encoder: BinaryEncoder, ordinal_encoder: OrdinalEncoder
) -> pd.DataFrame:
    """Aplica encoders YA AJUSTADOS y arma la matriz final de features.

    Requiere que `df` tenga las columnas de `SELECTED_FEATURES` (es decir, que
    ya haya pasado por `add_temporal_features`/`impute_missing`).
    """
    binary_encoded = binary_encoder.transform(df[BINARY_ENCODED_COLUMNS])
    cyclic = _add_cyclic_features(df)
    ordinal_encoded = pd.DataFrame(
        ordinal_encoder.transform(df[ORDINAL_ENCODED_COLUMNS]),
        columns=[f"ord_{c}" for c in ORDINAL_ENCODED_COLUMNS],
        index=df.index,
    )
    return pd.concat(
        [
            binary_encoded.reset_index(drop=True),
            cyclic.reset_index(drop=True),
            ordinal_encoded.reset_index(drop=True),
        ],
        axis=1,
    )


def build_features(
    raw_df: pd.DataFrame, binary_encoder: BinaryEncoder, ordinal_encoder: OrdinalEncoder
) -> pd.DataFrame:
    """Pipeline completo de inferencia: de columnas crudas a features codificadas.

    `raw_df` debe tener las columnas de `RAW_FEATURE_COLUMNS` (nombres tal cual
    el dataset original: `Location Description`, `Beat`, `Arrest`, `Domestic`,
    `Date`). Se usa tanto en el DAG de entrenamiento (fila a fila, en batch)
    como en `amq2/pipeline.py` para servir el modelo.
    """
    df = impute_missing(raw_df)
    df = add_temporal_features(df)
    return encode_features(df, binary_encoder, ordinal_encoder)
