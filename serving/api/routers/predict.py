"""Prediction endpoints."""

import io
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from serving.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HouseFeatures,
    PredictionResponse,
)
from src.housing_predictor.features.schema import API_TO_MODEL_FIELDS

router = APIRouter(prefix="/predict", tags=["Prediction"])


def _features_to_row(features):
    data = features.model_dump(by_alias=False)
    return {model_key: data[api_key] for api_key, model_key in API_TO_MODEL_FIELDS.items()}


def _get_top_features(pipeline):
    try:
        return pipeline.get_feature_importance(top_n=5).to_dict("records")
    except Exception:
        return None


@router.post("", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(features: HouseFeatures, request: Request):
    pipeline = request.app.state.inference_pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([_features_to_row(features)])
    preds, lower_bounds, upper_bounds = pipeline.predict_with_uncertainty(df)
    pred = float(preds[0])
    return PredictionResponse(
        prediction=pred,
        confidence_interval={
            "lower": float(pred + lower_bounds[0]),
            "upper": float(pred + upper_bounds[0]),
        },
        model_version="production",
        top_features=_get_top_features(pipeline),
    )


@router.post("/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK)
async def predict_batch(request_body: BatchPredictionRequest, request: Request):
    pipeline = request.app.state.inference_pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([_features_to_row(row) for row in request_body.features])
    preds, lower_bounds, upper_bounds = pipeline.predict_with_uncertainty(df)
    top_features = _get_top_features(pipeline)

    rows = []
    for idx in range(len(df)):
        pred = float(preds[idx])
        rows.append(
            PredictionResponse(
                prediction=pred,
                confidence_interval={
                    "lower": float(pred + lower_bounds[idx]),
                    "upper": float(pred + upper_bounds[idx]),
                },
                model_version="production",
                top_features=top_features,
            )
        )

    return BatchPredictionResponse(predictions=rows, total_processed=len(rows))


@router.post("/file", status_code=status.HTTP_200_OK)
async def predict_file(request: Request, file: UploadFile = File(...)):
    pipeline = request.app.state.inference_pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    filename = file.filename or "input"
    contents = await file.read()

    if filename.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(contents))
    elif filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(contents))
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    pipeline._validate_input(df)
    preds, lower_bounds, upper_bounds = pipeline.predict_with_uncertainty(df)

    out = df.copy()
    out["prediction"] = preds
    out["lower_bound"] = preds + lower_bounds
    out["upper_bound"] = preds + upper_bounds

    output = io.StringIO()
    out.to_csv(output, index=False)
    output.seek(0)

    headers = {"Content-Disposition": f'attachment; filename="{Path(filename).stem}_predictions.csv"'}
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers=headers)
