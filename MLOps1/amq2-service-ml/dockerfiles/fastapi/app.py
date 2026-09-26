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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Al arrancar, carga el modelo `MODEL_NAME@MODEL_ALIAS` desde el MLflow Model Registry."""
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    try:
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        model_state["model"] = mlflow.pyfunc.load_model(model_uri)
        client = MlflowClient()
        model_state["version"] = client.get_model_version_by_alias(
            MODEL_NAME, MODEL_ALIAS
        ).version
    except Exception:
        # Si todavía no se corrió `train_model`, no hay modelo registrado.
        # Se deja el servicio arriba (para healthcheck) y se falla recién en /predict.
        logger.exception("No se pudo cargar el modelo '%s@%s'", MODEL_NAME, MODEL_ALIAS)
        model_state["model"] = None
        model_state["version"] = None
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
    model = model_state["model"]
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No hay ningún modelo registrado como '{MODEL_NAME}@{MODEL_ALIAS}'. "
                "Correr los DAGs 'etl_process' y 'train_model' en Airflow primero."
            ),
        )

    case = pd.DataFrame([request.model_dump(by_alias=True)])
    prediction = model.predict(case)[0]

    return PredictResponse(
        prediction=str(prediction),
        model_name=MODEL_NAME,
        model_version=str(model_state["version"]),
    )
