"""Carga y curado del dataset de crímenes de Chicago.

Reproduce (con una simplificación: se descartan desde el inicio las columnas
que el notebook termina sin usar en las features finales, como `Community
Area`, `IUCR`, `Description` y `FBI Code`) el curado del notebook
`Crimenes_en_Chicago_Federico_Romero_Ailen_Muñoz.ipynb` del TP de Aprendizaje
de Máquina I: dataset de crímenes reportados en Chicago, clasificación
multiclase del `Primary Type` (tipo de crimen).

El dataset crudo se descarga desde la fuente oficial de City of Chicago y se
guarda en `s3://data/raw/reported_crimes.csv`. Si el objeto ya existe, el DAG
lo reutiliza para evitar bajar el archivo en cada corrida.
"""

from __future__ import annotations

import io
import os
import urllib.request

import boto3
from botocore.exceptions import ClientError
import pandas as pd
from sklearn.model_selection import train_test_split

from amq2.features import RAW_FEATURE_COLUMNS, TARGET_COLUMN
from amq2.storage import DATA_BUCKET, _s3_client

RAW_KEY = "raw/reported_crimes.csv"
RAW_DATASET_URL = os.environ.get(
    "RAW_DATASET_URL",
    "https://data.cityofchicago.org/api/views/9hwr-2zxp/rows.csv?accessType=DOWNLOAD",
)

_USECOLS = RAW_FEATURE_COLUMNS + [TARGET_COLUMN, "Case Number", "Updated On"]


def _parse_datetime_with_formats(values: pd.Series, formats: tuple[str, ...]) -> pd.Series:
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    for date_format in formats:
        missing = parsed.isna() & values.notna()
        if not missing.any():
            break
        parsed.loc[missing] = pd.to_datetime(
            values.loc[missing],
            format=date_format,
            errors="coerce",
        )
    return parsed


def _is_missing_s3_object(exc: ClientError) -> bool:
    status_code = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    error_code = exc.response.get("Error", {}).get("Code")
    return status_code == 404 or error_code in {"404", "NoSuchBucket", "NoSuchKey", "NotFound"}


def _ensure_data_bucket(client: boto3.client) -> None:
    try:
        client.head_bucket(Bucket=DATA_BUCKET)
    except ClientError as exc:
        if _is_missing_s3_object(exc):
            client.create_bucket(Bucket=DATA_BUCKET)
            return
        raise


def raw_dataset_exists(client: boto3.client | None = None) -> bool:
    """Indica si el CSV crudo ya existe en el bucket de datos."""
    client = client or _s3_client()
    try:
        obj = client.head_object(Bucket=DATA_BUCKET, Key=RAW_KEY)
    except ClientError as exc:
        if _is_missing_s3_object(exc):
            return False
        raise
    return int(obj.get("ContentLength", 0)) > 0


def download_raw_dataset(force: bool = False) -> str:
    """Descarga el CSV crudo y lo deja en MinIO si todavía no está disponible."""
    client: boto3.client = _s3_client()
    _ensure_data_bucket(client)
    if raw_dataset_exists(client) and not force:
        return f"s3://{DATA_BUCKET}/{RAW_KEY}"

    request = urllib.request.Request(
        RAW_DATASET_URL,
        headers={"User-Agent": "CEIA-MLOps1-Airflow/1.0"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        client.upload_fileobj(
            response,
            DATA_BUCKET,
            RAW_KEY,
            ExtraArgs={"ContentType": "text/csv"},
        )
    return f"s3://{DATA_BUCKET}/{RAW_KEY}"


def load_raw_data() -> pd.DataFrame:
    """Descarga el dataset crudo desde S3, lo retipa y elimina duplicados.

    Duplicados por `Case Number`: se conserva el registro con `Updated On` más
    reciente (idéntico criterio al notebook).
    """
    client: boto3.client = _s3_client()
    obj = client.get_object(Bucket=DATA_BUCKET, Key=RAW_KEY)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()), usecols=_USECOLS)

    df["Date"] = _parse_datetime_with_formats(
        df["Date"],
        ("%m/%d/%Y %I:%M:%S %p",),
    )
    df["Updated On"] = _parse_datetime_with_formats(
        df["Updated On"],
        ("%m/%d/%Y %I:%M:%S %p", "%Y %b %d %I:%M:%S %p"),
    )

    df = df.dropna(subset=["Case Number", "Updated On"])
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
