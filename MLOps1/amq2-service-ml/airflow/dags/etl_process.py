"""DAG de ETL: cura el dataset de crímenes de Chicago y genera train/test.

Reproduce el curado del TP de Aprendizaje de Máquina I (ver docstring de
`amq2/data.py` y `amq2/features.py`): dedup, split estratificado 80/20 (antes
de imputar/encodear, para evitar data leakage), imputación de
`Location Description`, features temporales cíclicas y Binary/Ordinal
Encoding ajustados solo con train.

Los encoders ajustados se guardan en `s3://data/processed/encoders.pkl` para
que el DAG `train_model` los empaquete junto con el modelo al servirlo.
"""

from __future__ import annotations

import pendulum
from airflow.sdk import dag, task

from amq2.features import (
    BINARY_ENCODED_COLUMNS,
    ORDINAL_ENCODED_COLUMNS,
    SELECTED_FEATURES,
    TARGET_COLUMN,
    add_temporal_features,
    build_binary_encoder,
    build_ordinal_encoder,
    encode_features,
    impute_missing,
)


@dag(
    dag_id="etl_process",
    description="ETL: cura el dataset de crímenes de Chicago y genera los splits de train/test.",
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=["amq2", "etl"],
)
def etl_process():
    """DAG: `load_and_split` -> `encode_and_save` (ver docstring del módulo)."""

    @task
    def load_and_split() -> dict[str, str]:
        """Carga el dataset crudo, hace el split y deja las features listas para encodear."""
        from amq2.data import load_raw_data, split_data
        from amq2.storage import save_dataframe

        df = load_raw_data()
        train_df, test_df = split_data(df)

        columns = SELECTED_FEATURES + [TARGET_COLUMN]
        train_df = add_temporal_features(impute_missing(train_df))[columns]
        test_df = add_temporal_features(impute_missing(test_df))[columns]

        save_dataframe(train_df, "train_clean.csv")
        save_dataframe(test_df, "test_clean.csv")
        return {"train": "train_clean.csv", "test": "test_clean.csv"}

    @task
    def encode_and_save(clean_keys: dict[str, str]) -> None:
        """Ajusta los encoders SOLO con train y guarda los datasets ya codificados."""
        import pickle

        from amq2.storage import load_dataframe, save_bytes, save_dataframe

        train_df = load_dataframe(clean_keys["train"])
        test_df = load_dataframe(clean_keys["test"])

        binary_encoder = build_binary_encoder().fit(train_df[BINARY_ENCODED_COLUMNS])
        ordinal_encoder = build_ordinal_encoder().fit(train_df[ORDINAL_ENCODED_COLUMNS])

        X_train = encode_features(train_df, binary_encoder, ordinal_encoder)
        X_test = encode_features(test_df, binary_encoder, ordinal_encoder)
        y_train = train_df[TARGET_COLUMN].reset_index(drop=True)
        y_test = test_df[TARGET_COLUMN].reset_index(drop=True)

        save_dataframe(X_train.assign(**{TARGET_COLUMN: y_train}), "train.csv")
        save_dataframe(X_test.assign(**{TARGET_COLUMN: y_test}), "test.csv")
        save_bytes(
            pickle.dumps({"binary_encoder": binary_encoder, "ordinal_encoder": ordinal_encoder}),
            "encoders.pkl",
        )

    encode_and_save(load_and_split())


etl_process()
