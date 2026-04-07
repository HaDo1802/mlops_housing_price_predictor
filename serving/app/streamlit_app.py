"""
Housing Price Prediction - Streamlit Web Application

This app provides a user-friendly interface for predicting housing prices
using the trained Gradient Boosting model.

Features:
- Interactive input forms for all required features
- Input validation
- Prediction with uncertainty intervals
- Feature importance visualization
- Professional result formatting

Usage:
    streamlit run serving/app/streamlit_app.py
"""

import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from predictor.schema import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    OPTIONAL_FEATURE_DEFAULTS,
)
from serving.api.feature_map import (
    CATEGORICAL_OPTIONS,
    FEATURE_DISPLAY_LABELS,
    VEGAS_DISTRICT_CENTROIDS,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")
DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/")


# Page configuration
st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
    }
    .interval-text {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-top: 0.5rem;
    }
    .feature-section {
        background-color: #f9f9f9;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ff4444;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e6f7ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1890ff;
        margin: 1rem 0;
    }
    </style>
""",
    unsafe_allow_html=True,
)


NUMERIC_INPUT_CONFIG = {
    "bedrooms": {"min_value": 0, "max_value": 20, "value": 3, "step": 1},
    "bathrooms": {"min_value": 0.0, "max_value": 20.0, "value": 2.0, "step": 0.5},
    "livingarea": {
        "min_value": 100.0,
        "max_value": 20000.0,
        "value": 1800.0,
        "step": 50.0,
    },
    "latitude": {"min_value": 35.5, "max_value": 36.5, "value": 36.1, "step": 0.0001},
    "longitude": {
        "min_value": -115.5,
        "max_value": -114.5,
        "value": -115.17,
        "step": 0.0001,
    },
}


@st.cache_data(ttl=60)
def fetch_model_info(base_url: str):
    """Fetch model metadata from the FastAPI service"""
    url = f"{base_url.rstrip('/')}/model/info"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        logger.warning(
            "Model info request failed: %s - %s", response.status_code, response.text
        )
    except requests.RequestException as exc:
        logger.warning("Model info request error: %s", exc)
    return None


@st.cache_data(ttl=60)
def fetch_model_schema(base_url: str):
    """Fetch canonical input feature contract from the API."""
    url = f"{base_url.rstrip('/')}/model/schema"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            payload = response.json()
            return payload.get("features", payload)
        logger.warning(
            "Model schema request failed: %s - %s", response.status_code, response.text
        )
    except requests.RequestException as exc:
        logger.warning("Model schema request error: %s", exc)
    return None


def resolve_feature_spec(
    schema_contract: Optional[dict],
) -> tuple[list[str], list[str], dict[str, str], dict[str, list[str]], list[str]]:
    """Resolve feature contract from API schema endpoint with local fallback."""
    numeric_features = list(NUMERIC_FEATURES)
    categorical_features = list(CATEGORICAL_FEATURES)
    display_labels = dict(FEATURE_DISPLAY_LABELS)
    categorical_options = dict(CATEGORICAL_OPTIONS)
    optional_features = list(OPTIONAL_FEATURE_DEFAULTS)

    if schema_contract:
        numeric_features = schema_contract.get("numeric") or numeric_features
        categorical_features = (
            schema_contract.get("categorical") or categorical_features
        )
        optional_features = schema_contract.get("optional") or optional_features
        display_labels.update(schema_contract.get("display_labels") or {})

        remote_options = schema_contract.get("categorical_options") or {}
        for key, values in remote_options.items():
            if isinstance(values, list) and values:
                categorical_options[key] = values

    return (
        numeric_features,
        categorical_features,
        display_labels,
        categorical_options,
        optional_features,
    )


@st.cache_data(ttl=30)
def fetch_api_health(base_url: str):
    """Fetch API health status for clearer diagnostics."""
    url = f"{base_url.rstrip('/')}/health"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        return None
    return None


def _api_base_url_candidates() -> list[str]:
    """Return candidate API base URLs ordered by runtime likelihood."""
    in_docker = Path("/.dockerenv").exists()
    candidates = []

    if in_docker:
        candidates.extend(["http://fastapi:8000", "http://localhost:8000"])
    else:
        candidates.extend(["http://localhost:8000", "http://127.0.0.1:8000"])

    if DEFAULT_API_BASE_URL:
        candidates.append(DEFAULT_API_BASE_URL.rstrip("/"))

    # Keep order, drop duplicates.
    deduped = []
    seen = set()
    for url in candidates:
        if url not in seen:
            deduped.append(url)
            seen.add(url)
    return deduped


@st.cache_data(ttl=30)
def resolve_api_base_url():
    """Find the first reachable API endpoint."""
    candidates = _api_base_url_candidates()
    for candidate in candidates:
        health = fetch_api_health(candidate)
        if health is not None:
            return candidate, candidates, health
    return None, candidates, None


def get_active_api_base_url() -> str:
    """Get currently selected API URL from session state."""
    return st.session_state.get("api_base_url") or DEFAULT_API_BASE_URL


def build_api_payload(inputs: dict) -> dict:
    """Return payload expected by current API schema keys."""
    payload = {k: v for k, v in inputs.items() if v is not None}
    if (
        ("latitude" not in payload or "longitude" not in payload)
        and "vegas_district" in payload
        and payload["vegas_district"] in VEGAS_DISTRICT_CENTROIDS
    ):
        centroid = VEGAS_DISTRICT_CENTROIDS[payload["vegas_district"]]
        payload.setdefault("latitude", float(centroid["latitude"]))
        payload.setdefault("longitude", float(centroid["longitude"]))
    return payload


def request_prediction(payload: dict) -> dict:
    """Call FastAPI prediction endpoint"""
    url = f"{get_active_api_base_url().rstrip('/')}/predict"
    try:
        response = requests.post(url, json=payload, timeout=20)
        if response.status_code == 200:
            return response.json()
        logger.error(
            "Prediction request failed: %s - %s", response.status_code, response.text
        )
        raise RuntimeError(response.text)
    except requests.RequestException as exc:
        logger.error("Prediction request error: %s", exc)
        raise RuntimeError("Prediction service is unavailable") from exc


def request_file_prediction(file_name: str, file_bytes: bytes, mime_type: str) -> bytes:
    """Call FastAPI file prediction endpoint and return CSV bytes"""
    url = f"{get_active_api_base_url().rstrip('/')}/predict/file"
    files = {"file": (file_name, file_bytes, mime_type or "application/octet-stream")}
    try:
        response = requests.post(url, files=files, timeout=60)
        if response.status_code == 200:
            return response.content
        logger.error(
            "File prediction request failed: %s - %s",
            response.status_code,
            response.text,
        )
        raise RuntimeError(response.text)
    except requests.RequestException as exc:
        logger.error("File prediction request error: %s", exc)
        raise RuntimeError("Prediction service is unavailable") from exc


def validate_inputs(inputs: dict, required_features: list) -> tuple:
    """
    Validate user inputs.

    Returns:
        (is_valid: bool, missing_fields: list, error_messages: list)
    """
    missing_fields = []
    error_messages = []

    # Check for missing fields
    for feature in required_features:
        if feature not in inputs or inputs[feature] is None or inputs[feature] == "":
            missing_fields.append(feature)

    # Check numeric ranges using per-feature input config.
    for feature, value in inputs.items():
        if feature in required_features:
            # Skip if already missing
            if feature in missing_fields:
                continue

            # Validate numeric ranges
            if isinstance(value, (int, float)):
                cfg = NUMERIC_INPUT_CONFIG.get(feature, {})
                min_value = cfg.get("min_value")
                max_value = cfg.get("max_value")
                if min_value is not None and value < min_value:
                    error_messages.append(f"{feature}: Must be >= {min_value}")
                if max_value is not None and value > max_value:
                    error_messages.append(f"{feature}: Must be <= {max_value}")

    is_valid = len(missing_fields) == 0 and len(error_messages) == 0

    return is_valid, missing_fields, error_messages


def create_input_form(
    numeric_features: list[str],
    categorical_features: list[str],
    display_labels: dict[str, str],
    categorical_options: dict[str, list[str]],
) -> tuple[dict, list[str]]:
    """Create input form from resolved feature specification."""

    st.markdown(
        '<div class="main-header">🏠 Housing Price Predictor</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Enter property details to predict the sale price</div>',
        unsafe_allow_html=True,
    )

    # Initialize session state for inputs
    if "inputs" not in st.session_state:
        st.session_state.inputs = {}

    inputs = {}
    st.subheader("Property Inputs")
    all_features = list(numeric_features) + list(categorical_features)
    if not all_features:
        st.info("No input features available from API/model metadata.")
        return inputs, all_features

    col_left, col_right = st.columns(2)
    for idx, feature in enumerate(all_features):
        col = col_left if idx % 2 == 0 else col_right
        with col:
            label = display_labels.get(feature, feature.replace("_", " ").title())
            if feature in numeric_features:
                params = NUMERIC_INPUT_CONFIG.get(
                    feature, {"min_value": 0.0, "value": 0.0, "step": 1.0}
                )
                inputs[feature] = st.number_input(
                    label,
                    key=f"input_{feature}",
                    help=f"Enter {label.lower()}",
                    **params,
                )
            else:
                options = categorical_options.get(feature)
                if options:
                    val = st.selectbox(
                        label,
                        options=[""] + options,
                        help=f"Select {label.lower()}",
                        key=f"input_{feature}",
                    )
                    inputs[feature] = None if val == "" else val
                else:
                    text = st.text_input(
                        label,
                        value="",
                        help=f"Enter {label.lower()}",
                        key=f"input_{feature}",
                    )
                    inputs[feature] = None if text.strip() == "" else text.strip()

    return inputs, numeric_features + categorical_features


def display_prediction_results(
    prediction, lower, upper, top_features, confidence_level: float = 95.0
):
    """Display prediction results in a professional format"""

    st.markdown("---")
    st.markdown("## 🎯 Prediction Results")

    # Main prediction box
    st.markdown(
        f"""
    <div class="prediction-box">
        <div class="prediction-value">${prediction:,.0f}</div>
        <div class="interval-text">
            Confidence Interval: ${lower:,.0f} - ${upper:,.0f}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Create two columns for additional info
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Prediction Statistics")

        # Calculate margin of error
        margin = (upper - lower) / 2
        margin_pct = (margin / prediction) * 100

        stats_df = pd.DataFrame(
            {
                "Metric": [
                    "Predicted Price",
                    "Lower Bound",
                    "Upper Bound",
                    "Margin of Error",
                    "Confidence Level",
                ],
                "Value": [
                    f"${prediction:,.0f}",
                    f"${lower:,.0f}",
                    f"${upper:,.0f}",
                    f"${margin:,.0f} (±{margin_pct:.1f}%)",
                    f"{confidence_level:.1f}%",
                ],
            }
        )

        st.dataframe(stats_df, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("### 🔑 Top 5 Most Important Features")

        if top_features is None or top_features.empty:
            st.info("Feature importance is not available from the API.")
        else:
            # Create horizontal bar chart for top features
            fig = go.Figure(
                go.Bar(
                    x=top_features["importance"].values,
                    y=top_features["feature"].values,
                    orientation="h",
                    marker=dict(
                        color=top_features["importance"].values,
                        colorscale="Blues",
                        showscale=False,
                    ),
                    text=top_features["importance"].apply(lambda x: f"{x:.4f}"),
                    textposition="auto",
                )
            )

            fig.update_layout(
                title="Feature Importance Scores",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis={"categoryorder": "total ascending"},
            )

            st.plotly_chart(fig, use_container_width=True)

    # Price range visualization
    st.markdown("### 📈 Price Range Visualization")

    fig = go.Figure()

    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=["Lower Bound", "Prediction", "Upper Bound"],
            y=[lower, prediction, upper],
            mode="markers+lines",
            marker=dict(size=[15, 25, 15], color=["#ff7f0e", "#1f77b4", "#ff7f0e"]),
            line=dict(color="#1f77b4", width=2),
            name="Price Range",
        )
    )

    # Add shaded area for confidence interval
    fig.add_trace(
        go.Scatter(
            x=["Lower Bound", "Prediction", "Upper Bound"],
            y=[lower, lower, lower],
            fill=None,
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=["Lower Bound", "Prediction", "Upper Bound"],
            y=[upper, upper, upper],
            fill="tonexty",
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            fillcolor="rgba(31, 119, 180, 0.2)",
            name="95% Confidence Interval",
        )
    )

    fig.update_layout(
        title="Predicted Price with Confidence Interval",
        xaxis_title="",
        yaxis_title="Price ($)",
        height=400,
        hovermode="x unified",
        yaxis=dict(tickformat="$,.0f"),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Add interpretation
    st.markdown(
        f"""
    <div class="info-box">
        <strong>💡 How to interpret these results:</strong><br>
        • The predicted sale price is <strong>${prediction:,.0f}</strong><br>
        • We are {confidence_level:.1f}% confident the actual price will be between <strong>${lower:,.0f}</strong> and <strong>${upper:,.0f}</strong><br>
        • The top 5 features shown above had the most influence on this prediction<br>
        • The margin of error is <strong>±${margin:,.0f}</strong> ({margin_pct:.1f}%)
    </div>
    """,
        unsafe_allow_html=True,
    )


def main():
    """Main application logic"""

    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1946/1946488.png", width=100)
        st.title("About")
        st.markdown(
            """
        This application predicts property prices using the deployed
        **Housing Predictor** model and a strict API feature contract.
        
        **Core input features:**
        - Bedrooms, Bathrooms, Living Area
        - Latitude, Longitude
        - Property Type
        
        **Instructions:**
        1. Fill in the required property details
        2. Click "Predict Price" button
        3. Review prediction with confidence interval
        4. Check which features influenced the prediction most
        """
        )

        st.markdown("**Input Requirements:**")
        st.markdown(
            f"""
        - `bedrooms`: {NUMERIC_INPUT_CONFIG['bedrooms']['min_value']} to {NUMERIC_INPUT_CONFIG['bedrooms']['max_value']}
        - `bathrooms`: {NUMERIC_INPUT_CONFIG['bathrooms']['min_value']} to {NUMERIC_INPUT_CONFIG['bathrooms']['max_value']}
        - `livingarea`: {NUMERIC_INPUT_CONFIG['livingarea']['min_value']} to {NUMERIC_INPUT_CONFIG['livingarea']['max_value']} sqft
        - `latitude` / `longitude`: optional if `vegas_district` is provided
        - `normalized_lot_area_value`: optional, defaults to {OPTIONAL_FEATURE_DEFAULTS['normalized_lot_area_value']:.0f} sqft when omitted
        - `propertytype`: select one of the provided categories
        """
        )

        st.markdown("---")
        st.markdown("**Model Info:**")
        resolved_url, candidates, health = resolve_api_base_url()
        if resolved_url:
            st.session_state.api_base_url = resolved_url

        default_sidebar_url = (
            st.session_state.get("api_base_url")
            or DEFAULT_API_BASE_URL
            or "http://localhost:8000"
        )
        api_base_url_input = st.text_input(
            "API Base URL",
            value=default_sidebar_url,
            help="Use your local FastAPI URL for development or your deployed API URL in Streamlit Cloud.",
        ).rstrip("/")
        if api_base_url_input:
            st.session_state.api_base_url = api_base_url_input

        active_url = get_active_api_base_url()
        health = fetch_api_health(active_url) if active_url else None
        st.caption(f"API URL: `{active_url}`")

        if health is None:
            st.error("API is unreachable. Start FastAPI locally or set a working `API_BASE_URL`.")
            st.code("uvicorn serving.api.main:app --reload --host 0.0.0.0 --port 8000")
            st.caption("Tried: " + ", ".join(f"`{url}`" for url in candidates))
            model_info = None
        elif not health.get("model_loaded", False):
            st.warning("API is running, but model is not loaded on startup.")
            load_error = health.get("load_error")
            if load_error:
                st.caption(f"Load error: {load_error}")
                if "invalid load key, 'v'" in str(load_error):
                    st.error(
                        "Model artifact looks like a Git LFS pointer file, not a real pickle. "
                        "Redeploy API with actual model binaries available at runtime."
                    )
            model_info = None
        else:
            model_info = fetch_model_info(active_url)
        if model_info:
            st.info(f"Model: {model_info.get('model_type', 'unknown')}")
            features = model_info.get("features", {}).get("count")
            if features is not None:
                st.info(f"Features: {features}")
            interval_cfg = model_info.get("prediction_interval") or {}
            st.session_state.confidence_level = float(
                interval_cfg.get("coverage", 0.95) * 100
            )
        else:
            st.warning("Model info unavailable. Using local fallback feature schema.")
            st.session_state.confidence_level = 95.0

        st.markdown("---")
        st.markdown("**Need Help?**")
        st.markdown(
            """
        - Bedrooms, bathrooms, living area, property type, and location are required
        - `normalized_lot_area_value` is optional and falls back to a default lot size
        - Numerical values must be within the allowed ranges shown in the form
        - Latitude/longitude should be in the target market bounds
        - Use exact categorical values shown in dropdowns
        """
        )

    # Main content
    schema_contract = fetch_model_schema(active_url) if active_url else None
    (
        numeric_features,
        categorical_features,
        display_labels,
        categorical_options,
        optional_features,
    ) = resolve_feature_spec(schema_contract)
    inputs, all_features = create_input_form(
        numeric_features,
        categorical_features,
        display_labels,
        categorical_options,
    )
    required_features = [f for f in all_features if f not in set(optional_features)]

    st.markdown("---")
    st.markdown("## 📂 Batch Predictions (CSV/Excel)")
    st.markdown(
        "Upload a CSV or Excel file with the same feature columns to get predictions for multiple rows."
    )

    uploaded_file = st.file_uploader(
        "Upload file",
        type=["csv", "xlsx", "xls"],
        help="Accepted formats: CSV, XLSX, XLS",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
        mime_type = uploaded_file.type or "application/octet-stream"

        st.info(f"Selected file: {file_name}")
        if st.button("🚀 Run Batch Prediction", type="primary"):
            with st.spinner("🔄 Processing file..."):
                try:
                    output_csv = request_file_prediction(
                        file_name=file_name,
                        file_bytes=file_bytes,
                        mime_type=mime_type,
                    )
                    output_name = f"{Path(file_name).stem}_predictions.csv"
                    st.success("✅ Batch predictions completed!")
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=output_csv,
                        file_name=output_name,
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"❌ Batch prediction failed: {str(e)}")
                    logger.error("Batch prediction error: %s", e, exc_info=True)

    # Predict button
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🔮 Predict Price", use_container_width=True, type="primary"
        )

    if predict_button:
        # Validate inputs
        is_valid, missing_fields, error_messages = validate_inputs(
            inputs, required_features
        )

        if not is_valid:
            # Display errors
            if missing_fields:
                st.markdown(
                    f"""
                <div class="error-message">
                    <strong>❌ Missing Required Fields:</strong><br>
                    {', '.join(missing_fields)}
                </div>
                """,
                    unsafe_allow_html=True,
                )

            if error_messages:
                st.markdown(
                    f"""
                <div class="error-message">
                    <strong>❌ Invalid Values:</strong><br>
                    {'<br>'.join(error_messages)}
                </div>
                """,
                    unsafe_allow_html=True,
                )

            st.warning(
                "⚠️ Please fill in all required fields with valid values before making a prediction."
            )

        else:
            # Make prediction
            with st.spinner("🔄 Calculating prediction..."):
                try:
                    payload = build_api_payload(inputs)
                    response = request_prediction(payload)

                    prediction = response["prediction"]
                    lower = response["confidence_interval"]["lower"]
                    upper = response["confidence_interval"]["upper"]
                    top_features = response.get("top_features")
                    if top_features is not None:
                        top_features = pd.DataFrame(top_features)

                    st.session_state.last_result = {
                        "prediction_id": str(uuid.uuid4()),
                        "prediction": float(prediction),
                        "lower": float(lower),
                        "upper": float(upper),
                        "top_features": top_features,
                    }
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
                    logger.error(f"Prediction error: {str(e)}", exc_info=True)

    if "last_result" in st.session_state:
        result = st.session_state.last_result

        # Display results
        display_prediction_results(
            result["prediction"],
            result["lower"],
            result["upper"],
            result["top_features"],
            st.session_state.get("confidence_level", 95.0),
        )

        # Success message
        st.success("✅ Prediction completed successfully!")

        # Download button for results
        results_data = {
            "Predicted Price": [f"${result['prediction']:,.0f}"],
            "Lower Bound (95% CI)": [f"${result['lower']:,.0f}"],
            "Upper Bound (95% CI)": [f"${result['upper']:,.0f}"],
            "Margin of Error": [f"${(result['upper'] - result['lower']) / 2:,.0f}"],
        }
        results_df = pd.DataFrame(results_data)

        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Prediction Results",
            data=csv,
            file_name="house_price_prediction.csv",
            mime="text/csv",
        )

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <small>Housing Price Predictor v1.0 | Built with Streamlit | 
        Powered by Gradient Boosting ML Model</small>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
