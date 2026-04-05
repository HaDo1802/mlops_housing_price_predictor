"""Training feature schema constants."""

NUMERIC_FEATURES = [
    "bedrooms",
    "bathrooms",
    "living_area",
    "latitude",
    "longitude",
    "normalized_lot_area_value",
    "days_on_zillow",
]

CATEGORICAL_FEATURES = [
    "property_type",
    "vegas_district",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_COLUMN = "price"

DROP_COLUMNS = [
    "property_id",
    "snapshot_date",
    "street_address",
    "city",
    "state",
    "zip_code",  # 8 missing, weaker signal than vegas_district + lat/lon
    "zestimate",  # 100% null
    "rentzestimate",  # 100% null
    "listing_status",  # 100% FOR_SALE — zero variance
    "normalized_lot_area_unit",  # 100% sqft — zero variance
    "price_per_sqft",  # derived from price/living_area — data leakage
]

OUTLIER_FILTERS = {
    "price_per_sqft_min": 80.0,  # below this = data error or land-only
    "price_per_sqft_max": 800.0,  # above this = ultra-luxury, distorts model
    "living_area_min": 400.0,  # below this = unrealistic
    "bedrooms_max": 10,  # above this = mansion/multi-unit, small sample
    "bathrooms_max": 10,
}

EXCLUDED_PROPERTY_TYPES = [
    "MOBILE",  # priced off lot-lease, not sqft — median $124k vs $490k overall
]
