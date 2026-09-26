"""Carga y curado del dataset de crímenes de Chicago.

Reproduce (con una simplificación: se descartan desde el inicio las columnas
que el notebook termina sin usar en las features finales, como `Community
Area`, `IUCR`, `Description` y `FBI Code`) el curado del notebook
`Crimenes_en_Chicago_Federico_Romero_Ailen_Muñoz.ipynb` del TP de Aprendizaje
de Máquina I: dataset de crímenes reportados en Chicago, clasificación
multiclase del `Primary Type` (tipo de crimen).

El dataset crudo se sube una vez a `s3://data/raw/reported_crimes.csv` (no se
descarga desde Google Drive en cada corrida del DAG, para no depender de una
fuente externa en cada ejecución).
"""

from __future__ import annotations

import io

import boto3
import pandas as pd
from sklearn.model_selection import train_test_split

from amq2.features import RAW_FEATURE_COLUMNS, TARGET_COLUMN
from amq2.storage import DATA_BUCKET, _s3_client

RAW_KEY = "raw/reported_crimes.csv"

_USECOLS = RAW_FEATURE_COLUMNS + [TARGET_COLUMN, "Case Number", "Updated On"]


def load_raw_data() -> pd.DataFrame:
    """Descarga el dataset crudo desde S3, lo retipa y elimina duplicados.

    Duplicados por `Case Number`: se conserva el registro con `Updated On` más
    reciente (idéntico criterio al notebook).
    """
    client: boto3.client = _s3_client()
    obj = client.get_object(Bucket=DATA_BUCKET, Key=RAW_KEY)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()), usecols=_USECOLS)

    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    df["Updated On"] = pd.to_datetime(
        df["Updated On"], format="%Y %b %d %I:%M:%S %p", errors="coerce"
    )

    df = df.loc[df.groupby("Case Number")["Updated On"].idxmax()]
    return df.drop(columns=["Case Number", "Updated On"])


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split 80/20 estratificado por `Primary Type` (igual que el notebook)."""
    return train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET_COLUMN],
    )
