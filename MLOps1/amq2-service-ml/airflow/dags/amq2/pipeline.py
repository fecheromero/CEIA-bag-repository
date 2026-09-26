"""Wrapper de MLflow que empaqueta encoders + clasificador como un solo modelo servible.

Se loguea con `mlflow.pyfunc.log_model(..., code_path=[<carpeta amq2>])`, así el
mismo código (`amq2.features`) queda embebido en el artefacto del modelo y es
importable también desde el contenedor de FastAPI (que no tiene acceso al
código de Airflow). `predict()` recibe un DataFrame con las columnas crudas
(`Location Description`, `Beat`, `Arrest`, `Domestic`, `Date`) y devuelve la
clase de crimen predicha (`Primary Type`), sin que quien llame necesite saber
nada sobre el encoding interno.
"""

from __future__ import annotations

import pickle

import mlflow.pyfunc
import pandas as pd

from amq2.features import RAW_FEATURE_COLUMNS, build_features


class ChicagoCrimeModel(mlflow.pyfunc.PythonModel):
    """Modelo servible: encoders (fit en entrenamiento) + clasificador Random Forest."""

    def load_context(self, context):
        """Deserializa encoders y clasificador desde los artefactos logueados con el modelo."""
        with open(context.artifacts["encoders"], "rb") as f:
            encoders = pickle.load(f)
        self.binary_encoder = encoders["binary_encoder"]
        self.ordinal_encoder = encoders["ordinal_encoder"]
        with open(context.artifacts["classifier"], "rb") as f:
            self.classifier = pickle.load(f)

    def predict(self, context, model_input: pd.DataFrame, params=None):
        """Encodea las columnas crudas de `model_input` y predice `Primary Type`."""
        model_input = model_input[RAW_FEATURE_COLUMNS]
        features = build_features(model_input, self.binary_encoder, self.ordinal_encoder)
        return self.classifier.predict(features)
