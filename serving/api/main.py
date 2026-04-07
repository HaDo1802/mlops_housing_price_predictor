"""FastAPI entrypoint."""

import logging
import os
from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from serving.api.routers.health import router as health_router
from serving.api.routers.model import router as model_router
from serving.api.routers.predict import router as predict_router
from predictor.predict import InferencePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _mask_bucket_name(bucket: str | None) -> str | None:
    if not bucket:
        return None
    if len(bucket) <= 4:
        return "*" * len(bucket)
    return f"{bucket[:2]}***{bucket[-2:]}"

app = FastAPI(title="Housing Price Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(model_router)
app.include_router(predict_router)


@app.get("/")
async def root():
    return {
        "message": "Housing Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.on_event("startup")
async def startup_event():
    model_name = "housing_price_predictor"
    stage_candidates = ["Production", "Staging"]
    app.state.model_load_error = None

    for stage in stage_candidates:
        try:
            app.state.inference_pipeline = InferencePipeline(
                model_name=model_name, stage=stage
            )
            bucket = _mask_bucket_name(os.getenv("ARTIFACT_BUCKET"))
            model_type = app.state.inference_pipeline.metadata.get("model_type")
            logger.info(
                "Inference pipeline loaded (model=%s, stage=%s, source=%s, bucket=%s, model_type=%s)",
                model_name,
                stage,
                app.state.inference_pipeline._loaded_from,
                bucket,
                model_type,
            )
            return
        except Exception as exc:
            app.state.model_load_error = str(exc)
            logger.warning(
                "Failed to load pipeline (model=%s, stage=%s): %s",
                model_name,
                stage,
                exc,
            )

    app.state.inference_pipeline = None
    logger.error(
        "Failed to load inference pipeline for all stage candidates: %s",
        stage_candidates,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})
