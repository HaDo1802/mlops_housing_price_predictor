"""FastAPI entrypoint."""

import logging
from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from serving.api.routers.health import router as health_router
from serving.api.routers.model import router as model_router
from serving.api.routers.predict import router as predict_router
from src.housing_predictor.pipelines.inference import InferencePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    return {"message": "Housing Price Prediction API", "version": "1.0.0", "docs": "/docs"}


@app.on_event("startup")
async def startup_event():
    try:
        app.state.inference_pipeline = InferencePipeline(
            model_name="housing_price_predictor", stage="Production"
        )
        logger.info("Inference pipeline loaded")
    except Exception as exc:
        logger.error("Failed to load pipeline: %s", exc)
        app.state.inference_pipeline = None


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})
