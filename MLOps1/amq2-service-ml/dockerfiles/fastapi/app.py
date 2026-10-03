"""API REST para servir el modelo de clasificación de crímenes de Chicago (AMq1).

El modelo se carga desde el MLflow Model Registry usando `MODEL_NAME` y
`MODEL_ALIAS` (ver `.env`): un `mlflow.pyfunc` (`amq2.pipeline.ChicagoCrimeModel`)
que empaqueta encoders + Random Forest, generado por los DAGs `etl_process` +
`train_model` de Airflow. Este módulo no conoce el encoding interno del
modelo, solo sabe "cargar lo que esté registrado con ese nombre/alias" y
pasarle el DataFrame crudo del request.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient

from schemas import PredictRequest, PredictResponse

logger = logging.getLogger("uvicorn.error")

MODEL_NAME = os.environ.get("MODEL_NAME", "amq2_model")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")

model_state: dict[str, Any] = {"model": None, "version": None}


def refresh_model_if_needed() -> Any:
    """Carga el modelo champion si falta o si el alias cambió de versión."""
    client = MlflowClient()
    try:
        model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    except Exception as exc:
        model_state["model"] = None
        model_state["version"] = None
        raise HTTPException(
            status_code=503,
            detail=(
                f"No hay ningún modelo registrado como '{MODEL_NAME}@{MODEL_ALIAS}'. "
                "Correr los DAGs 'etl_process' y 'train_model' en Airflow primero."
            ),
        ) from exc

    version = str(model_version.version)
    if model_state["model"] is None or model_state["version"] != version:
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        logger.info("Cargando modelo '%s@%s' version %s", MODEL_NAME, MODEL_ALIAS, version)
        try:
            model = mlflow.pyfunc.load_model(model_uri)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"No se pudo cargar el modelo '{MODEL_NAME}@{MODEL_ALIAS}' version {version}.",
            ) from exc
        model_state["model"] = model
        model_state["version"] = version

    return model_state["model"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configura MLflow y precarga el modelo si ya existe en el Model Registry."""
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    try:
        refresh_model_if_needed()
    except Exception:
        # Si todavía no se corrió `train_model`, no hay modelo registrado.
        # Se deja el servicio arriba (para healthcheck) y se falla recién en /predict.
        logger.exception("No se pudo cargar el modelo '%s@%s'", MODEL_NAME, MODEL_ALIAS)
    yield


app = FastAPI(
    title="ML Models and something more Inc. - Model Service",
    description="Sirve el modelo del TP de MLOps1 registrado en MLflow.",
    lifespan=lifespan,
)


@app.get("/")
def read_root():
    """Healthcheck del servicio."""
    return {"message": "Welcome to the Model Service"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predice el tipo de crimen (`Primary Type`) para el incidente recibido.

    El modelo servido es el registrado en MLflow bajo `MODEL_NAME` con el
    alias `MODEL_ALIAS`: un `mlflow.pyfunc` que aplica el mismo feature
    engineering/encoding del entrenamiento (ver `amq2/pipeline.py`) antes de
    predecir, generado por los DAGs `etl_process` + `train_model` de Airflow.
    """
    model = refresh_model_if_needed()

    case = pd.DataFrame([request.model_dump(by_alias=True)])
    prediction = model.predict(case)[0]

    return PredictResponse(
        prediction=str(prediction),
        model_name=MODEL_NAME,
        model_version=str(model_state["version"]),
    )
