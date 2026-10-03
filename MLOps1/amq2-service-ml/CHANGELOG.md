# Changelog

Este archivo se genera automaticamente desde el historial de Git.
Para actualizarlo, ejecutar:

```bash
python3 scripts/generate_changelog.py
```

## Features

- 2026-10-03 `87927bb` mlops1: descarga dataset crudo en dag
- 2026-09-26 `00958fd` fastapi: carga el modelo champion y expone POST /predict
- 2026-09-26 `9061d8e` fastapi: schema PredictRequest con los campos crudos del incidente
- 2026-09-26 `901668a` airflow: DAG train_model (búsqueda en MLflow y registro como champion)
- 2026-09-26 `c46e24c` airflow: DAG etl_process (curado y splits en S3)
- 2026-09-26 `40fb0e7` amq2: wrapper mlflow.pyfunc que empaqueta los encoders y el clasificador
- 2026-09-26 `7cd3809` amq2: Random Forest con grid search de hiperparámetros
- 2026-09-26 `29f5315` amq2: feature engineering y encoding fiel al notebook de AMq1
- 2026-09-26 `ce69967` amq2: carga y dedup del dataset crudo de crímenes de Chicago
- 2026-09-26 `4791bea` amq2: helpers de lectura/escritura en MinIO

## Fixes

- 2026-10-03 `8891dfa` mlops1: usa psycopg2 para backend de mlflow
- 2026-10-03 `2f4746c` fastapi: recarga modelo champion en predict
- 2026-10-03 `1e81ad8` mlops1: fija mlflow y driver postgres

## Documentacion

- 2026-10-03 `39d2572` mlops1: aclara pasos de despliegue
- 2026-10-03 `a75be03` mlops1: actualiza changelog
- 2026-10-03 `a2d23c3` mlops1: actualiza changelog
- 2026-10-03 `8d72620` mlops1: agrega changelog e integrantes
- 2026-09-26 `1037009` mlops1: reescribe el README (despliegue, prerrequisitos, nota sobre las métricas)
- 2026-09-26 `aef47cf` mlops1: agrega ARQUITECTURA.md
- 2026-09-26 `1d3c24d` mlops1: agrega los datasets curados de train/test
- 2026-09-26 `59fdcef` mlops1: agrega el notebook original de AMq1

## Build

- 2026-10-03 `862f562` mlops1: agrega psutil al entorno de serving
- 2026-10-03 `12ac02a` mlops1: alinea dependencias de entrenamiento y serving
- 2026-10-03 `76c3e2c` mlops1: fija tags de imagenes docker
- 2026-09-26 `e47e6de` fastapi: agrega mlflow, boto3, pandas, scikit-learn y category_encoders
- 2026-09-26 `bbd0a5a` airflow: agrega scikit-learn, pandas y category_encoders

## Mantenimiento

- 2026-09-26 `f6cd8aa` airflow: desactiva los DAGs de ejemplo (LOAD_EXAMPLES)
- 2026-09-26 `c731a56` mlops1: agrega MODEL_NAME/MODEL_ALIAS y MLFLOW_TRACKING_URI al compose
- 2026-09-26 `dae12da` mlops1: remapea puertos de MinIO (9010/9011) y MLflow (5011)
- 2026-09-23 `feea749` iniciar TP de MLOps1 con template amq2-service-ml

## Otros cambios

- 2026-09-27 `553dc90` subo post de ejemplo
