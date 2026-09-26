"""DAG de entrenamiento: búsqueda de hiperparámetros y registro del modelo servible.

Corre `GridSearchCV` sobre Random Forest (la familia de modelo elegida en el
TP de Aprendizaje de Máquina I, ver `amq2/model.py`) logueando cada
combinación como un run anidado en MLflow, evalúa el mejor modelo en test con
las mismas métricas del notebook (accuracy, exactitud balanceada y F1 macro)
y lo registra en el Model Registry como un `mlflow.pyfunc` que empaqueta el
clasificador junto con los encoders de `etl_process` (ver `amq2/pipeline.py`),
de forma que quien sirva el modelo no necesite conocer el encoding interno.
"""

from __future__ import annotations

import os

import pendulum
from airflow.sdk import dag, task

from amq2.features import TARGET_COLUMN
from amq2.model import get_estimator_and_param_grid

EXPERIMENT_NAME = "amq2_hyperparameter_search"
MODEL_NAME = os.environ.get("MODEL_NAME", "amq2_model")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")


@dag(
    dag_id="train_model",
    description="Búsqueda de hiperparámetros en MLflow y registro del modelo servible.",
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=["amq2", "training", "mlflow"],
)
def train_model():
    """DAG: `hyperparameter_search` -> `register_best_model` (ver docstring del módulo)."""

    @task
    def hyperparameter_search() -> str:
        """Busca hiperparámetros en una submuestra y reentrena el ganador con todo el train.

        Random Forest sobre ~190k filas x 25 features x 31 clases desbalanceadas
        puede generar árboles grandes; correr el `GridSearchCV` completo sobre
        todo `train.csv` agotó la memoria disponible en el entorno local usado
        para probar este pipeline. Se busca sobre una submuestra estratificada
        (más liviana) y el modelo final se reentrena con el dataset completo,
        que es una práctica habitual cuando el cómputo es limitado.
        """
        import mlflow
        from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

        from amq2.storage import load_dataframe

        train_df = load_dataframe("train.csv")
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        y_train = train_df[TARGET_COLUMN]

        SEARCH_SAMPLE_SIZE = 40_000
        X_search, _, y_search, _ = train_test_split(
            X_train, y_train,
            train_size=SEARCH_SAMPLE_SIZE,
            random_state=42,
            stratify=y_train,
        )

        estimator, param_grid = get_estimator_and_param_grid()
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name="grid_search") as parent_run:
            mlflow.log_param("search_sample_size", len(X_search))
            search = GridSearchCV(estimator, param_grid, cv=cv, scoring="f1_macro", refit=False)
            search.fit(X_search, y_search)

            for params, mean_score in zip(
                search.cv_results_["params"], search.cv_results_["mean_test_score"]
            ):
                with mlflow.start_run(nested=True):
                    mlflow.log_params(params)
                    mlflow.log_metric("mean_cv_f1_macro", mean_score)

            mlflow.log_params(search.best_params_)
            mlflow.log_metric("best_cv_f1_macro", search.best_score_)

            best_estimator, _ = get_estimator_and_param_grid()
            best_estimator.set_params(**search.best_params_)
            best_estimator.fit(X_train, y_train)
            mlflow.sklearn.log_model(best_estimator, "classifier")

            return parent_run.info.run_id

    @task
    def register_best_model(parent_run_id: str) -> None:
        """Evalúa el mejor modelo en test y registra el pyfunc servible (encoders + modelo)."""
        import pickle
        import tempfile
        from pathlib import Path

        import mlflow
        from mlflow import MlflowClient
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

        from amq2.pipeline import ChicagoCrimeModel
        from amq2.storage import load_bytes, load_dataframe

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

        test_df = load_dataframe("test.csv")
        X_test = test_df.drop(columns=[TARGET_COLUMN])
        y_test = test_df[TARGET_COLUMN]

        classifier = mlflow.sklearn.load_model(f"runs:/{parent_run_id}/classifier")
        y_pred = classifier.predict(X_test)

        with mlflow.start_run(run_id=parent_run_id):
            mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
            mlflow.log_metric(
                "test_balanced_accuracy", balanced_accuracy_score(y_test, y_pred)
            )
            mlflow.log_metric("test_f1_macro", f1_score(y_test, y_pred, average="macro"))

            with tempfile.TemporaryDirectory() as tmp_dir:
                encoders_path = Path(tmp_dir) / "encoders.pkl"
                classifier_path = Path(tmp_dir) / "classifier.pkl"
                encoders_path.write_bytes(load_bytes("encoders.pkl"))
                classifier_path.write_bytes(pickle.dumps(classifier))

                amq2_package_dir = str(Path(__file__).resolve().parent / "amq2")
                model_info = mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=ChicagoCrimeModel(),
                    artifacts={
                        "encoders": str(encoders_path),
                        "classifier": str(classifier_path),
                    },
                    code_paths=[amq2_package_dir],
                )

        client = MlflowClient()
        registered = mlflow.register_model(model_info.model_uri, MODEL_NAME)
        client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, registered.version)

    register_best_model(hyperparameter_search())


train_model()
