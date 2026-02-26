"""Model information endpoints."""

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/model", tags=["Model"])


@router.get("/info")
async def model_info(request: Request):
    pipeline = request.app.state.inference_pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": pipeline.metadata.get("model_type"),
        "hyperparameters": pipeline.metadata.get("hyperparameters"),
        "metrics": {
            "test": pipeline.metadata.get("test_metrics"),
            "validation": pipeline.metadata.get("val_metrics"),
        },
        "features": {
            "count": len(pipeline.metadata.get("feature_names", [])),
            "names": pipeline.metadata.get("feature_names", []),
        },
        "training_info": {
            "train_size": pipeline.metadata.get("train_size"),
            "val_size": pipeline.metadata.get("val_size"),
            "test_size": pipeline.metadata.get("test_size"),
        },
    }
