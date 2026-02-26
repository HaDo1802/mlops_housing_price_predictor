"""Pydantic schemas for the inference API."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class HouseFeatures(BaseModel):
    lot_area: float = Field(..., gt=0)
    total_bsmt_sf: float = Field(..., ge=0)
    first_flr_sf: float = Field(..., gt=0, alias="1st Flr SF")
    second_flr_sf: float = Field(..., ge=0, alias="2nd Flr SF")
    gr_liv_area: float = Field(..., gt=0)
    garage_area: float = Field(..., ge=0)
    overall_qual: int = Field(..., ge=1, le=10)
    overall_cond: int = Field(..., ge=1, le=10)
    year_built: int = Field(..., ge=1800, le=2100)
    year_remod_add: int = Field(..., ge=1800, le=2100)
    bedroom_abvgr: int = Field(..., ge=0, le=10)
    full_bath: int = Field(..., ge=0, le=5)
    half_bath: int = Field(..., ge=0, le=3)
    totrms_abvgrd: int = Field(..., ge=0, le=20)
    fireplaces: int = Field(..., ge=0, le=5)
    garage_cars: int = Field(..., ge=0, le=5)
    neighborhood: str
    ms_zoning: str
    bldg_type: str
    house_style: str
    foundation: str
    central_air: str = Field(..., pattern="^[NY]$")
    garage_type: str

    @field_validator("year_remod_add")
    @classmethod
    def validate_remod_year(cls, v: int, info):
        year_built = info.data.get("year_built")
        if year_built is not None and v < year_built:
            raise ValueError("Remodel year cannot be before build year")
        return v


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
