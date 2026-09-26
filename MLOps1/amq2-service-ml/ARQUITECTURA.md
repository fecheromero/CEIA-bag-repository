# Arquitectura

Este documento describe los componentes reales que forman el sistema desplegado por
`docker-compose.yaml` para el TP de MLOps1, sus puertos, su función, qué contienen
internamente (DAGs, tareas, módulos) y cómo se comunican entre sí. El diagrama de alto
nivel del template original está en `final_assign.png` (ver `README.md`); este documento
profundiza en la configuración real, incluyendo el modelo de Aprendizaje de Máquina I ya
integrado (clasificación de `Primary Type` sobre crímenes de Chicago).

## 1. Vista general

```
                         ┌─────────────────────────────────────────────┐
                         │                  Cliente                     │
                         │   (browser / curl / notebook / awscli)       │
                         └───────┬───────────┬───────────┬─────────────┘
                                 │           │           │
                     :8080       │  :5011    │   :9011   │   :8800
                  (Airflow UI)   │ (MLflow UI)│ (MinIO UI)│ (FastAPI)
                                 ▼           ▼           ▼
        ┌────────────────────────────┐ ┌──────────┐ ┌─────────┐ ┌───────────────┐
        │  Airflow (5 procesos)      │ │  MLflow  │ │  MinIO  │ │    FastAPI    │
        │  apiserver/scheduler/      │ │ tracking │ │  (S3)   │ │  (serving)    │
        │  dag-processor/worker/     │ │ + registry│ │         │ │               │
        │  triggerer                 │ └────┬─────┘ └────┬────┘ └───────┬───────┘
        └──────┬───────────┬────────┘      │            │              │
               │           │                │            │              │
       (Celery)│           │ boto3          │ mlflow     │ boto3        │ mlflow.pyfunc
               ▼           ▼                ▼ client     ▼ (artifacts)  ▼ .load_model
        ┌──────────┐ ┌───────────────────────────────────────────────────────┐
        │  Redis   │ │                     Postgres                          │
        │ (broker) │ │  DB `airflow` (metadata Airflow) · DB `mlflow_db`     │
        └──────────┘ │             (tracking store de MLflow)                │
                      └───────────────────────────────────────────────────────┘
```

Todos los servicios corren en dos redes Docker (`frontend`, `backend`, ambas bridge
simples definidas al final de `docker-compose.yaml`). Postgres y Redis solo están en
`backend`; el resto de los servicios (Airflow, MLflow, MinIO, FastAPI) están en ambas, así
que en la práctica `backend` es la red donde vive todo el estado (DB, cola) y por donde
todos los servicios llegan a él.

## 2. Tabla de componentes

| Componente | Contenedor(es) | Puerto host → interno | Red(es) | Rol |
|---|---|---|---|---|
| Airflow API Server | `airflow-apiserver` | `8080 → 8080` | frontend, backend | UI web + REST API de Airflow |
| Airflow Scheduler | `amq2-service-ml-airflow-scheduler-1` | – | frontend, backend | Decide cuándo correr cada DAG/tarea |
| Airflow DAG Processor | `amq2-service-ml-airflow-dag-processor-1` | – | frontend, backend | Parsea los `.py` de `airflow/dags/` y los serializa a la DB |
| Airflow Worker | `amq2-service-ml-airflow-worker-1` | – | frontend, backend | Ejecuta las tareas (Celery worker, concurrencia 2) |
| Airflow Triggerer | `amq2-service-ml-airflow-triggerer-1` | – | frontend, backend | Maneja tareas *deferred*/sensores async |
| Airflow Init / CLI | `amq2-service-ml-airflow-init-1` / `airflow_cli` | – | frontend, backend | Setup inicial (migraciones, usuario admin) / debugging manual (`--profile debug`) |
| MLflow | `mlflow` | `5011 → 5000` (via `MLFLOW_PORT`) | frontend, backend | Tracking server + Model Registry |
| MinIO (S3) | `minio` | `9010→9000` API, `9011→9001` consola (via `MINIO_PORT`/`MINIO_PORT_UI`) | frontend, backend | Object storage (data lake + artifact store de MLflow) |
| `create_s3_buckets` | `minio_create_bucket` | – | backend | Job one-shot: crea los buckets `data` y `mlflow` al levantar |
| Postgres | `postgres` | `5432 → 5432` | backend | DB relacional: metadata de Airflow + backend store de MLflow |
| Redis (Valkey) | (sin `container_name`, servicio `redis`) | `6379` (solo `expose`, no publicado al host) | backend | Broker + result backend de Celery para Airflow |
| FastAPI | `fastapi` | `8800 → 8800` | frontend, backend | Sirve el modelo registrado (`POST /predict`) |

Los puertos por defecto salen de `.env`; varios ya fueron remapeados en este entorno
porque 9000/5001 estaban ocupados por otros procesos locales (ver `README.md`).

## 3. Persistencia (dónde vive cada cosa)

| Dato | Dónde | Detalle |
|---|---|---|
| Metadata de Airflow (DAGs, runs, tareas, usuarios, conexiones) | Postgres, DB `airflow` | Volumen `db_data` |
| Cola de tareas Celery + resultados | Redis | En memoria, sin volumen (efímero) |
| Runs/experimentos/métricas/parámetros de MLflow | Postgres, DB `mlflow_db` | `--backend-store-uri postgresql://.../mlflow_db` |
| Artefactos de MLflow (modelos serializados, pickles, encoders) | MinIO, bucket `mlflow` | `--default-artifact-root s3://mlflow/` |
| Dataset crudo | MinIO, bucket `data`, prefijo `raw/` | `reported_crimes.csv` (subido una vez a mano) |
| Dataset curado (train/test, con y sin encodear) + encoders ajustados | MinIO, bucket `data`, prefijo `processed/` | `train_clean.csv`, `test_clean.csv`, `train.csv`, `test.csv`, `encoders.pkl` |
| Object storage físico de MinIO | Volumen `minio_data` | Contiene ambos buckets |
| Variables/conexiones de Airflow versionadas en código | `airflow/secrets/{variables,connections}.yaml` | Montado como `/opt/secrets`, backend `LocalFilesystemBackend` |
| Modelo registrado (Model Registry) | MLflow (metadata en Postgres, artefactos en MinIO) | Nombre `amq2_model`, alias `champion` apuntando a una versión |

No hay ninguna base de datos NoSQL/documental en este stack: todo el estado
estructurado (Airflow + MLflow) vive en el mismo Postgres, en bases separadas.

## 4. Airflow: DAGs y sus tareas

Ambos DAGs están en `airflow/dags/` y comparten el paquete `airflow/dags/amq2/`
(módulos puros, sin dependencia de Airflow, reutilizables entre DAGs y por el modelo
servido). `AIRFLOW__CORE__LOAD_EXAMPLES=false`, así que solo estos 2 DAGs aparecen en
la UI.

### 4.1 `etl_process.py`

Cura el dataset de crímenes de Chicago (dedup, split estratificado, imputación,
features temporales, encoding) — ver docstring del archivo para el detalle fiel al
notebook de AMq1.

| Tarea | Responsabilidad | Lee | Escribe |
|---|---|---|---|
| `load_and_split` | Carga el CSV crudo, dedup por `Case Number`, split 80/20 estratificado, imputa `Location Description`, deriva features temporales cíclicas | `s3://data/raw/reported_crimes.csv` | `s3://data/processed/{train,test}_clean.csv` |
| `encode_and_save` | Ajusta Binary/Ordinal Encoder **solo con train**, encodea train y test, persiste los encoders | `train_clean.csv`, `test_clean.csv` | `train.csv`, `test.csv`, `encoders.pkl` |

### 4.2 `train_model.py`

Busca hiperparámetros de un Random Forest y registra el modelo servible.

| Tarea | Responsabilidad | Lee | Escribe |
|---|---|---|---|
| `hyperparameter_search` | `GridSearchCV` (Random Forest, `cv=3`, `scoring=f1_macro`) sobre una submuestra estratificada de 40k filas (la búsqueda sobre el dataset completo agotaba la memoria local); reentrena el mejor combo con el train completo | `train.csv` | Runs anidados + modelo sklearn en MLflow (experimento `amq2_hyperparameter_search`) |
| `register_best_model` | Evalúa en test (`accuracy`, `balanced_accuracy`, `f1_macro`), empaqueta encoders + clasificador en un `mlflow.pyfunc` (`ChicagoCrimeModel`) y lo registra | `test.csv`, `encoders.pkl`, modelo de la tarea anterior | Modelo registrado `amq2_model`, alias `champion` |

### 4.3 Paquete compartido `airflow/dags/amq2/`

| Módulo | Responsabilidad |
|---|---|
| `data.py` | Acceso al dataset crudo en S3, dedup, split train/test |
| `features.py` | Feature engineering + encoding puro (temporal cíclico, Binary/Ordinal Encoding); **sin I/O**, para poder empaquetarse junto al modelo |
| `model.py` | Estimador base (Random Forest) y grilla de hiperparámetros a explorar |
| `pipeline.py` | `ChicagoCrimeModel(mlflow.pyfunc.PythonModel)`: wrapper que en `predict()` aplica `features.py` y luego el clasificador — es lo que queda registrado como `amq2_model` |
| `storage.py` | Helpers de lectura/escritura a S3 (CSV y bytes crudos) sobre el bucket `data` |

## 5. MLflow

- **Tracking server**: contenedor `mlflow`, expuesto en `:5011` (host) → `:5000`
  (contenedor). Los clientes (Airflow, FastAPI) le hablan por la red interna en
  `http://mlflow:5000` (env var `MLFLOW_TRACKING_URI`).
- **Backend store** (experimentos, runs, params, métricas, versiones del registry):
  Postgres, DB `mlflow_db`.
- **Artifact store** (archivos: modelos serializados, `encoders.pkl`, `classifier`
  pickled): bucket `mlflow` en MinIO, vía `MLFLOW_S3_ENDPOINT_URL=http://s3:9000`.
- **Experimento**: `amq2_hyperparameter_search` — un run padre `grid_search` por
  ejecución de `train_model`, con runs anidados por combinación de hiperparámetros.
- **Model Registry**: modelo `amq2_model`; el alias `champion` apunta siempre a la
  última versión registrada por `train_model`. Es un `mlflow.pyfunc` (no un modelo
  sklearn plano), por lo que `mlflow.pyfunc.load_model("models:/amq2_model@champion")`
  ya incluye el encoding interno.

## 6. FastAPI (serving)

Código en `dockerfiles/fastapi/`.

| Archivo | Responsabilidad |
|---|---|
| `app.py` | App FastAPI. En el `lifespan` (startup) carga `models:/{MODEL_NAME}@{MODEL_ALIAS}` desde MLflow. Expone `GET /` (healthcheck) y `POST /predict` |
| `schemas.py` | `PredictRequest`/`PredictResponse` — los campos crudos de un incidente (`Location Description`, `Beat`, `Arrest`, `Domestic`, `Date`), no las features codificadas |

FastAPI no sabe nada de encoders ni de las 25 columnas del modelo: le pasa el
DataFrame crudo al `mlflow.pyfunc` cargado, que hace todo el feature engineering
internamente (mismo código de `amq2/features.py`, embebido en el artefacto del modelo
vía `code_paths` al loguearlo).

## 7. Flujo de comunicación de punta a punta

1. **Carga inicial**: se sube `reported_crimes.csv` a `s3://data/raw/` (manual, una
   sola vez).
2. **ETL**: Airflow Worker corre `etl_process` → lee de MinIO (boto3) → escribe
   datasets curados/encodeados + `encoders.pkl` de vuelta a MinIO.
3. **Entrenamiento**: Airflow Worker corre `train_model` → lee `train.csv`/`test.csv`
   de MinIO → entrena, y por cada run llama a MLflow (`http://mlflow:5000`) para
   loguear params/métricas/artefactos → MLflow persiste metadata en Postgres
   (`mlflow_db`) y los archivos en MinIO (bucket `mlflow`) → al final registra la
   versión y mueve el alias `champion`.
4. **Serving**: al arrancar, FastAPI le pide a MLflow (`http://mlflow:5000`) la
   versión con alias `champion`; MLflow resuelve los artefactos desde MinIO y
   FastAPI los descarga y carga en memoria.
5. **Inferencia**: un cliente HTTP pega a `POST http://localhost:8800/predict`;
   FastAPI arma un DataFrame con los campos crudos, se lo pasa al `pyfunc` cargado
   (que encodea + predice in-process, sin más llamadas de red) y devuelve la
   predicción.
6. **Orquestación interna de Airflow**: Scheduler decide qué correr → encola en
   Redis → Worker (Celery) la toma y ejecuta → resultado/estado va a Postgres (DB
   `airflow`) → DAG Processor mantiene sincronizada la definición de los DAGs desde
   `airflow/dags/*.py` hacia esa misma base.
