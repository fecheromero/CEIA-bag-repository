"""Helpers para leer/escribir los datasets procesados en el bucket de datos (MinIO/S3)."""

from __future__ import annotations

import io
import os

import boto3
import pandas as pd

DATA_BUCKET = os.environ.get("DATA_REPO_BUCKET_NAME", "data")
PROCESSED_PREFIX = "processed"


def _s3_client():
    """Cliente boto3 apuntando al endpoint S3 de MinIO (credenciales por env vars)."""
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3", "http://s3:9000"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )


def save_dataframe(df: pd.DataFrame, key: str) -> None:
    """Sube un DataFrame como CSV a `s3://{DATA_BUCKET}/{PROCESSED_PREFIX}/{key}`."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    _s3_client().put_object(
        Bucket=DATA_BUCKET,
        Key=f"{PROCESSED_PREFIX}/{key}",
        Body=buffer.getvalue().encode("utf-8"),
    )


def load_dataframe(key: str) -> pd.DataFrame:
    """Descarga y parsea un CSV desde `s3://{DATA_BUCKET}/{PROCESSED_PREFIX}/{key}`."""
    obj = _s3_client().get_object(Bucket=DATA_BUCKET, Key=f"{PROCESSED_PREFIX}/{key}")
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def save_bytes(data: bytes, key: str) -> None:
    """Sube bytes crudos a `s3://{DATA_BUCKET}/{PROCESSED_PREFIX}/{key}` (para pickles)."""
    _s3_client().put_object(Bucket=DATA_BUCKET, Key=f"{PROCESSED_PREFIX}/{key}", Body=data)


def load_bytes(key: str) -> bytes:
    """Descarga bytes crudos desde `s3://{DATA_BUCKET}/{PROCESSED_PREFIX}/{key}`."""
    obj = _s3_client().get_object(Bucket=DATA_BUCKET, Key=f"{PROCESSED_PREFIX}/{key}")
    return obj["Body"].read()
