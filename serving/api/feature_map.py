"""API-facing feature mappings and display metadata."""

API_TO_MODEL_FIELDS = {
    "bedrooms": "bedrooms",
    "bathrooms": "bathrooms",
    "living_area": "living_area",
    "livingarea": "living_area",  # backward compat for old API clients
    "latitude": "latitude",
    "longitude": "longitude",
    "normalized_lot_area_value": "normalized_lot_area_value",
    "days_on_zillow": "days_on_zillow",
    "property_type": "property_type",
    "propertytype": "property_type",  # backward compat
    "vegas_district": "vegas_district",
}

FEATURE_DISPLAY_LABELS = {
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "living_area": "Living Area (sqft)",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "normalized_lot_area_value": "Lot Size (sqft)",
    "days_on_zillow": "Days on Market",
    "property_type": "Property Type",
    "vegas_district": "District",
}

CATEGORICAL_OPTIONS = {
    "property_type": [
        "SINGLE_FAMILY",
        "TOWNHOUSE",
        "CONDO",
        "MULTI_FAMILY",
    ],
    "vegas_district": [
        "Summerlin",
        "Centennial",
        "Winchester",
        "Spring Valley",
        "Enterprise",
        "Mountains Edge",
        "Downtown Las Vegas",
        "Paradise",
        "Green Valley",
        "North Las Vegas",
        "The Strip",
        "Anthem",
    ],
}

VEGAS_DISTRICT_CENTROIDS = {
    "Summerlin": {"latitude": 36.1699, "longitude": -115.3378},
    "Green Valley": {"latitude": 36.0417, "longitude": -115.0833},
    "Henderson": {"latitude": 36.0395, "longitude": -114.9817},
    "Downtown Las Vegas": {"latitude": 36.1699, "longitude": -115.1398},
    "North Las Vegas": {"latitude": 36.1989, "longitude": -115.1175},
    "Spring Valley": {"latitude": 36.1021, "longitude": -115.2450},
    "Paradise": {"latitude": 36.0972, "longitude": -115.1467},
    "Enterprise": {"latitude": 36.0253, "longitude": -115.2419},
    "Centennial": {"latitude": 36.2839, "longitude": -115.2710},
    "Mountains Edge": {"latitude": 36.0089, "longitude": -115.2747},
    "The Strip": {"latitude": 36.1147, "longitude": -115.1728},
    "Winchester": {"latitude": 36.1420, "longitude": -115.0987},
    "Anthem": {"latitude": 35.9853, "longitude": -115.1017},
}
