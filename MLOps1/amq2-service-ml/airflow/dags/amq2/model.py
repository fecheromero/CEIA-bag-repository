"""Estimador y grilla de hiperparámetros para la búsqueda en MLflow.

El TP de Aprendizaje de Máquina I comparó árbol de decisión, bagging, Random
Forest y soft voting sobre la clasificación de `Primary Type`, y se quedó con
**Random Forest** (mejor F1 macro y exactitud balanceada) con hiperparámetros
fijos (`n_estimators=200`, `class_weight='balanced'`). Acá mantenemos esa
familia de modelo pero la sometemos a una búsqueda de hiperparámetros real
(a diferencia del notebook, que no tuneó el Random Forest), que es lo que
pide este TP de MLOps1.

La grilla se mantiene chica a propósito: con ~190k filas x 25 features y 31
clases con asociación débil a las features (según el propio notebook), un
Random Forest tiende a construir árboles enormes en cantidad de nodos.
Probando este pipeline, ni acotar `max_depth` a 20 alcanzó (el reentrenamiento
sobre el train completo agotó más de 14GB de RAM), así que además de acotar
`max_depth` se fija `max_leaf_nodes`, que limita el tamaño del árbol de forma
directa sin depender de qué tan desbalanceados/ruidosos resulten los splits.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier


def get_estimator_and_param_grid() -> tuple[Any, dict[str, list[Any]]]:
    """Devuelve el estimador base y la grilla de hiperparámetros a explorar."""
    estimator = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=2,
        max_leaf_nodes=2000,
    )
    param_grid = {
        "n_estimators": [100, 150],
        "max_depth": [10, 20],
    }
    return estimator, param_grid
