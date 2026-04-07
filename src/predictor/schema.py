"""Training feature schema constants."""

NUMERIC_FEATURES = [
    "bedrooms",
    "bathrooms",
    "living_area",
    "latitude",
    "longitude",
    "normalized_lot_area_value",
]

CATEGORICAL_FEATURES = [
    "property_type",
    "vegas_district",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

OPTIONAL_FEATURE_DEFAULTS = {
    "normalized_lot_area_value": 5000.0,
}

TARGET_COLUMN = "price"

DROP_COLUMNS = [
    "property_id",
    "snapshot_date",
    "street_address",
    "city",
    "state",
    "zip_code",
    "zestimate",
    "rentzestimate",
    "listing_status",
    "normalized_lot_area_unit",
    "days_on_zillow",
    "price_per_sqft",
]
