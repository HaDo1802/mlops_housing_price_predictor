"""Pydantic schemas for the inference API."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class HouseFeatures(BaseModel):
    bedrooms: int = Field(..., ge=0, le=20)
    bathrooms: float = Field(..., ge=0, le=20)
    livingarea: float = Field(..., gt=0)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    normalized_lot_area_value: Optional[float] = None
    days_on_zillow: Optional[float] = None
    propertytype: str
    vegas_district: Optional[str] = None


class PredictionResponse(BaseModel):
    prediction: float
    confidence_interval: Dict[str, float]
    model_version: str
    top_features: Optional[List[FeatureImportance]] = None


class BatchPredictionRequest(BaseModel):
    features: List[HouseFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    load_error: Optional[str] = None
