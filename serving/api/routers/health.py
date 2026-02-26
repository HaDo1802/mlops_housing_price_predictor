"""Health endpoints."""

from fastapi import APIRouter, Request

from serving.api.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    pipeline = request.app.state.inference_pipeline
    loaded = pipeline is not None
    version = pipeline.metadata.get("model_type", "unknown") if loaded else None
    return HealthResponse(
        status="healthy" if loaded else "unhealthy",
        model_loaded=loaded,
        model_version=version,
    )
