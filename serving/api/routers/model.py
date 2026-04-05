"""Model information endpoints."""

from fastapi import APIRouter, HTTPException, Request

from housing_predictor.features.training_schema import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from serving.api.feature_map import CATEGORICAL_OPTIONS, FEATURE_DISPLAY_LABELS

router = APIRouter(prefix="/model", tags=["Model"])


def _raw_feature_names(pipeline) -> list:
    """Best-effort raw input feature list (before one-hot expansion)."""
    preprocessor = getattr(pipeline, "preprocessor", None)
    if preprocessor is None:
        return []

    numeric = getattr(preprocessor, "numeric_features", None) or []
    categorical = getattr(preprocessor, "categorical_features", None) or []
    return list(numeric) + list(categorical)


@router.get("/info")
async def model_info(request: Request):
    pipeline = request.app.state.inference_pipeline
    if pipeline is None:
        load_error = getattr(request.app.state, "model_load_error", None)
        detail = "Model not loaded"
        if load_error:
            detail = f"{detail}: {load_error}"
        raise HTTPException(status_code=503, detail=detail)

    transformed_feature_names = pipeline.metadata.get("feature_names", [])
    # Use canonical contract as single source of truth for API clients.
    raw_feature_names = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    artifact_raw_feature_names = _raw_feature_names(pipeline)

    return {
        "model_type": pipeline.metadata.get("model_type"),
        "hyperparameters": pipeline.metadata.get("hyperparameters"),
        "metrics": {
            "test": pipeline.metadata.get("test_metrics"),
            "validation": pipeline.metadata.get("val_metrics"),
        },
        "prediction_interval": pipeline.metadata.get("prediction_interval"),
        "features": {
            "count": len(raw_feature_names),
            "names": raw_feature_names,
        },
        "artifact_features": {
            "count": len(artifact_raw_feature_names),
            "names": artifact_raw_feature_names,
        },
        "transformed_features": {
            "count": len(transformed_feature_names),
            "names": transformed_feature_names,
        },
        "training_info": {
            "train_size": pipeline.metadata.get("train_size"),
            "val_size": pipeline.metadata.get("val_size"),
            "test_size": pipeline.metadata.get("test_size"),
        },
    }


@router.get("/schema")
async def model_schema():
    """Canonical raw input feature contract used by API and Streamlit."""
    feature_names = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    return {
        "features": {
            "required": feature_names,
            "numeric": list(NUMERIC_FEATURES),
            "categorical": list(CATEGORICAL_FEATURES),
            "display_labels": FEATURE_DISPLAY_LABELS,
            "categorical_options": CATEGORICAL_OPTIONS,
        }
    }
