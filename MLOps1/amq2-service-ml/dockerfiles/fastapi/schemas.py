"""Schemas de entrada/salida del endpoint `/predict`.

Coinciden con las columnas crudas del dataset de crímenes de Chicago
(TP de Aprendizaje de Máquina I). El modelo servido (un `mlflow.pyfunc` que
empaqueta encoders + Random Forest, ver `amq2/pipeline.py` en el DAG de
entrenamiento) se encarga de todo el feature engineering/encoding interno, así
que este schema solo pide datos "de caso" tal como los tendría quien reporta
un incidente.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Datos de un incidente a partir de los cuales se predice el tipo de crimen."""

    location_description: str = Field(
        ...,
        alias="Location Description",
        description="Descripción del lugar del incidente (ej. 'STREET', 'RESIDENCE')",
        examples=["STREET"],
    )
    beat: int = Field(
        ..., alias="Beat", description="Beat (subdivisión policial) donde ocurrió el incidente"
    )
    arrest: bool = Field(..., alias="Arrest", description="¿Hubo arresto?")
    domestic: bool = Field(..., alias="Domestic", description="¿Fue un incidente doméstico?")
    incident_datetime: datetime = Field(
        ...,
        alias="Date",
        description="Fecha y hora del incidente (se derivan día de la semana, mes y franja horaria)",
        examples=["2023-07-14T23:30:00"],
    )

    model_config = {"populate_by_name": True}


class PredictResponse(BaseModel):
    """Resultado de la predicción."""

    prediction: str = Field(..., description="Tipo de crimen predicho (`Primary Type`)")
    model_name: str = Field(..., description="Nombre del modelo servido")
    model_version: str = Field(..., description="Versión del modelo servido en el Model Registry")
