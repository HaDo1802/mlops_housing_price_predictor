[![ml-pipeline-ci](https://github.com/HaDo1802/housing_price_predictor/actions/workflows/ml_pipeline_ci.yml/badge.svg)](https://github.com/HaDo1802/housing_price_predictor/actions/workflows/ml_pipeline_ci.yml)

# Housing Price Predictor: End-to-End MLOps Project 

![Project Cover](image/architecture.png)


- Streamlit UI: [`vegas-housing-price-predictor`](https://vegas-house-price-predict.streamlit.app/)


Transforming the classic **beginner house price prediction** problem into a **production-grade machine learning project** that implements practical MLOps patterns across the entire lifecycle:
- config-oriented management using config.ymal 
- modular data + feature pipelines
- reproducible training and evaluation
- experiment tracking and model governance with MLflow
- conditional promotion to production
- FastAPI for local/backend development and Streamlit for deployment
- drift checks (PSI-based)
- CI quality gates

## Airflow Orchestration

The project now includes two Airflow DAGs to automate training operations:

- `data_ingestion_dag`: runs daily ingestion, then drift gate logic, then triggers retraining.
- `retraining_dag`: runs retraining steps and writes training artifacts.

### DAG: Data Ingestion + Drift Gate

![Data Ingestion DAG](image/data_drift_dag.png)

### DAG: Retraining Pipeline

![Retraining DAG](image/retrain_dag.png)

Current operating mode:

- We are intentionally retraining daily for now to populate training history quickly.
- This is acceptable because current training time is fast, and I did not registry training model automatically .
- PSI drift detection logic is already implemented and can be used as the strict gate once dataset volume grows.



## Project Goals

This repository is designed as a learning + portfolio project to show how an ML model can move from notebook experimentation into a maintainable production workflow.

Core goals:

- Build a reliable training pipeline with leakage-safe preprocessing.
- Track runs, metrics, and lineage with MLflow.
- Promote models using explicit quality gates instead of manual intuition.
- Serve predictions through a stable API and web UI.
- Keep deployment paths flexible for local and containerized runtime.
- Add monitoring hooks for post-deployment feedback and drift detection.

## Why This Architecture

The codebase is intentionally separated by responsibility:

- `src/predictor/`: core ML package logic (reusable + testable).
- `scripts/`: operational entrypoints for jobs (train, promote, sync artifacts).
- `serving/`: online inference layer (FastAPI + Streamlit).
- `conf/`: single-file configuration in `conf/config.yaml`.
- `tests/`: unit + integration tests to protect behavior.

This structure scales better than notebook-centric projects because each concern evolves independently:

- model logic changes do not require API rewrites
- deployment/runtime changes do not require training rewrites
- config changes do not require code edits

## MLOps Practices Implemented

### 1) Single-File Configuration

Implemented in [config.py](src/predictor/config.py) with a single YAML source:

- `conf/config.yaml`

Why this pattern matters:

- keeps training configuration explicit and easy to audit
- avoids hard-coding paths and hyperparameters
- improves reproducibility and auditability

### 2) Leakage-Safe Data and Feature Pipeline

Implemented across:

- [data_ingest.py](src/predictor/data_ingest.py)
- [preprocessor.py](src/predictor/preprocessor.py)
- [training_pipeline.py](src/predictor/training_pipeline.py)

Key practice:

- split first
- fit preprocessing only on training data
- transform val/test/production with the fitted transformer

Why this pattern matters:

- prevents target/data leakage
- keeps offline evaluation closer to production reality
- ensures consistent feature transformations online/offline

### 3) Reproducible Model Factory + Validation

Implemented in [models.py](src/predictor/models.py):

- model type registry (`random_forest`, `ridge`, `gradient_boosting`, `hist_gradient_boosting`), as user can choose mulitple models for testing purpose
- hyperparameter validation against sklearn constructor signatures
- optional `TransformedTargetRegressor` (`log1p`/`expm1`) for stable target modeling

Why this pattern matters:

- limits configuration mistakes
- standardizes model creation
- makes experiments comparable
- serve as a centralized remote that control all the variable/config for the project

### 4) Experiment Tracking and Registry Governance

Implemented in:

- [training_pipeline.py](src/predictor/training_pipeline.py)
- [registry.py](src/predictor/registry.py)

Tracked in MLflow:

- parameters, metrics, tags, model artifact, feature metadata, config snapshots
- model version tags including git commit and model type

Why this pattern matters:

- you can answer "what model is in production and where did it come from?"
- supports rollback/debugging and lineage traceability

### 5) Metric Gate Before Promotion

Implemented in [registry.py](src/predictor/registry.py):

- compare candidate metric vs current production metric
- promote only if threshold is exceeded

Default behavior:

- if no production model exists: accept candidate
- else require improvement (default threshold `0.02` on `test_r2`)

Why this pattern matters:

- prevents accidental regressions from being promoted
- turns model promotion into a policy, not an ad-hoc decision

### 6) Artifact Strategy for Serving Reliability

Implemented in [sync_production_artifacts.py](scripts/sync_production_artifacts.py), [artifact_store.py](src/predictor/artifact_store.py), and [predict.py](src/predictor/predict.py):

- sync the current Production model into a stable artifact snapshot
- publish the production snapshot to S3
- production inference loads from the S3 production snapshot
- explicit local artifact loading is kept only for tests and local debugging

Why this pattern matters:

- keeps serving decoupled from MLflow runtime availability
- gives Streamlit and future APIs one stable production artifact source

### 7) Online Serving Interface

Implemented in:

- [main.py](serving/api/main.py)
- [predict.py](serving/api/routers/predict.py)
- [model.py](serving/api/routers/model.py)
- [health.py](serving/api/routers/health.py)
- [streamlit_app.py](serving/app/streamlit_app.py)

Serving capabilities:

- single prediction
- batch prediction
- file upload prediction (`.csv`/Excel)
- health endpoint
- model info endpoint
- confidence interval output (when estimator supports ensembles)

Why this pattern matters:

- clean separation between model internals and consumer-facing interfaces
- easier integration with product frontends and external services

Current serving mode:

- [streamlit_app.py](serving/app/streamlit_app.py) is the primary deployed interface
- FastAPI remains in the repo so other engineers can still run `uvicorn serving.api.main:app --reload` locally and build their own API-based integrations
- both Streamlit and FastAPI use the same inference layer in [predict.py](src/predictor/predict.py)
- production inference expects S3 artifact configuration via `ARTIFACT_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_DEFAULT_REGION`

### 8) Post-Deployment Drift Monitoring

Implemented in:

- [drift.py](src/predictor/drift.py)

Current monitoring includes:

- PSI drift checks against a reference snapshot

Why this pattern matters:

- shifts project from pure training to lifecycle monitoring
- creates a path toward retraining triggers and model observability

### 9) CI Quality Gates

Implemented in [ml_pipeline_ci.yml](.github/workflows/ml_pipeline_ci.yml):

- formatting check (`black --check`)
- linting (`flake8`)
- tests (`pytest`)

Why this pattern matters:

- enforces consistent standards before merge
- catches integration errors early (e.g., missing tracked modules/imports)

## End-to-End Workflow

```text
Raw data
  -> schema-driven feature contract
  -> train/val/test split
  -> fit preprocessor on train only
  -> train model
  -> evaluate on val/test
  -> log run + artifacts to MLflow
  -> select candidate run
  -> register/promote (if pass)
  -> sync Production artifacts to local + S3
  -> serve via Streamlit/API
  -> monitor drift
```

## Repository Structure

```text
.
├── conf/                        # Single YAML configuration
├── data/                        # Raw, processed, sample, and feedback datasets
├── docker/                      # Dockerfiles + compose setup
├── image/                       # Cover image and media assets
├── notebooks/                   # Exploration and experimentation notebooks
├── scripts/                     # CLI/job scripts (train, promote, sync artifacts)
├── serving/                     # FastAPI service + Streamlit app
├── src/predictor/               # Core ML package
├── tests/                       # Unit + integration tests
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Quick Start

### 1) Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 2) Run Quality Checks

```bash
make format
make lint
make test
```

### 3) Train Model Pipeline

```bash
python scripts/train.py
```

### 4) Inspect MLflow

```bash
mlflow ui
```

Open `http://localhost:5000`.

### 5) Serve FastAPI

```bash
make api
```

FastAPI is kept for local development and future integrations. With the current production inference setup, make sure these environment variables are available before starting the API:

```bash
export ARTIFACT_BUCKET=your-bucket-name
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-west-1
```

### 6) Run Streamlit UI

```bash
make ui
```

This is the primary deployed interface. Streamlit also expects the same S3 artifact environment variables when running against the production model snapshot.

## Pipeline Scripts

- Train pipeline:

```bash
python scripts/train.py
```

- Promote model / sync production artifacts:

```bash
python scripts/promote.py --model-name housing_price_predictor --version 1 --stage Production
python scripts/sync_production_artifacts.py
```

- Drift check utility:

```bash
python -m predictor.drift
```

## API Endpoints

Base URL local: `http://localhost:8000`

- `GET /health`
- `GET /model/info`
- `POST /predict`
- `POST /predict/batch`
- `POST /predict/file`

Swagger docs: `http://localhost:8000/docs`

## Docker Deployment

```bash
docker compose -f docker/docker-compose.yml up --build
```

Services:

- FastAPI on `8000`
- Streamlit on `8501`

`models/production` is mounted read-only into containers for local artifact workflows. The primary production serving path loads from the S3 production snapshot.

## Testing Strategy

Tests currently include:

- training-to-inference contract checks
- unit tests for config loading
- unit tests for data cleaning and training-column selection
- unit tests for preprocessor fit/transform behavior
- unit tests for registry governance
- unit tests for production artifact sync
- unit tests for regression metric outputs

Run:

```bash
python -m pytest tests -v
```

## Scaling and Future Evolution

The current design already supports several scaling directions:

- More models:
  Add new estimators in the trainer registry without changing serving contract.
- More environments:
  Use config layering to separate local/CI/prod data paths and settings.
- Stronger governance:
  Extend promotion gates with multiple metrics and safety checks.
- Better observability:
  Push feedback/drift metrics to a dashboard or alerting stack.
- Safer deployments:
  Add canary/staging traffic split using model stage transitions.

## Learning Outcomes Demonstrated

This project demonstrates practical skills in:

- ML system design and modular architecture
- reproducible training pipelines
- experiment tracking and model registry workflows
- model promotion governance using metric thresholds
- production API and UI integration
- monitoring-aware ML lifecycle design
- CI/CD quality automation
- containerized deployment patterns

## Author

Ha Do
- Email: havando1802@gmail.com
