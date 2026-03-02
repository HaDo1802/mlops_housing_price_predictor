"""Simple drift monitoring utilities for housing price data."""

from datetime import date
import json
import logging
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd

try:
    from data.query_data_from_supabase import load_dashboard_df
except ModuleNotFoundError:
    # Allow running this file directly via:
    # python src/housing_predictor/monitoring/drift.py
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    sys.path.append(str(PROJECT_ROOT))
    from data.query_data_from_supabase import load_dashboard_df
logger = logging.getLogger(__name__)

DEFAULT_REFERENCE_PATH = Path("data/snapshot/drift_reference.json")
DEFAULT_PSI_BINS = 10
PSI_THRESHOLD_MODERATE = 0.01
PSI_THRESHOLD_HIGH = 0.02

PRICE_COL = "price"
DATE_COL = "snapshot_date"


def compute_psi(
    reference: pd.Series,
    current: pd.Series,
    bins: int = DEFAULT_PSI_BINS,
) -> Optional[float]:
    """Compute PSI between two numeric series."""
    reference = reference.dropna()
    current = current.dropna()

    if len(reference) < 10 or len(current) < 10:
        logger.warning(
            "PSI skipped: insufficient data (reference=%d, current=%d).",
            len(reference),
            len(current),
        )
        return None

    if reference.nunique() < 2:
        logger.warning("PSI skipped: reference series is constant.")
        return None

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference.values, quantiles))

    if len(edges) < 3:
        logger.warning("PSI skipped: too few unique quantile edges.")
        return None

    ref_counts, _ = np.histogram(reference.values, bins=edges)
    cur_counts, _ = np.histogram(current.values, bins=edges)

    eps = 1e-6
    ref_pct = np.where(ref_counts == 0, eps, ref_counts / ref_counts.sum())
    cur_pct = np.where(cur_counts == 0, eps, cur_counts / cur_counts.sum())

    psi = float(np.sum((ref_pct - cur_pct) * np.log(ref_pct / cur_pct)))
    return psi


def compute_price_psi(
    reference_prices: pd.Series,
    current_prices: pd.Series,
    bins: int = DEFAULT_PSI_BINS,
) -> Optional[float]:
    """Compute PSI on log-transformed prices."""
    ref_log = np.log1p(reference_prices.dropna())
    cur_log = np.log1p(current_prices.dropna())
    return compute_psi(pd.Series(ref_log), pd.Series(cur_log), bins=bins)


def save_reference_snapshot(
    df: pd.DataFrame,
    reference_path: Path = DEFAULT_REFERENCE_PATH,
    bins: int = DEFAULT_PSI_BINS,
) -> dict:
    """Save summary stats used as drift reference."""
    reference_path = Path(reference_path)
    reference_path.parent.mkdir(parents=True, exist_ok=True)

    prices = df[PRICE_COL].dropna()
    log_prices = np.log1p(prices.values)
    if DATE_COL in df.columns:
        snapshot_date = str(pd.to_datetime(df[DATE_COL]).max().date())
    else:
        snapshot_date = "unknown"

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(log_prices, quantiles))
    ref_counts, _ = np.histogram(log_prices, bins=edges)
    ref_pct = ref_counts / ref_counts.sum()
    raw_quantiles = np.quantile(prices.dropna().values, np.linspace(0, 1, bins + 1))

    snapshot = {
        "snapshot_date": snapshot_date,
        "row_count": int(len(df)),
        "bins": bins,
        "price_median": float(prices.median()),
        "price_mean": float(prices.mean()),
        "price_std": float(prices.std()),
        "price_log_bin_edges": [float(e) for e in edges],
        "price_log_ref_pct": [float(p) for p in ref_pct],
        "price_raw_quantiles": [float(v) for v in raw_quantiles],
    }

    with open(reference_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    logger.info("Reference snapshot saved to %s (rows=%d).", reference_path, len(df))
    return snapshot


def load_reference_snapshot(
    reference_path: Path = DEFAULT_REFERENCE_PATH,
) -> Optional[dict]:
    """Load saved drift reference. Return None on first run."""
    reference_path = Path(reference_path)
    if not reference_path.exists():
        logger.info(
            "No reference snapshot found at %s. This is expected on the first run.",
            reference_path,
        )
        return None

    with open(reference_path, "r") as f:
        snapshot = json.load(f)

    logger.info(
        "Reference snapshot loaded (date=%s, rows=%d).",
        snapshot.get("snapshot_date"),
        snapshot.get("row_count"),
    )
    return snapshot


def run_drift_check(
    current_df: pd.DataFrame,
    reference_path: Path = DEFAULT_REFERENCE_PATH,
    psi_threshold: float = PSI_THRESHOLD_MODERATE,
    bins: int = DEFAULT_PSI_BINS,
) -> dict:
    """Compare current data to saved reference and decide if retraining is needed."""
    if PRICE_COL not in current_df.columns:
        raise ValueError(
            f"Column '{PRICE_COL}' not found in current_df. "
            f"Available columns: {list(current_df.columns)}"
        )

    current_prices = current_df[PRICE_COL].dropna()
    current_median = float(current_prices.median())
    current_row_count = int(len(current_df))
    if DATE_COL in current_df.columns:
        current_date = str(pd.to_datetime(current_df[DATE_COL]).max().date())
    else:
        current_date = "unknown"

    reference = load_reference_snapshot(reference_path)

    if reference is None:
        logger.info("No reference snapshot found. Current data will become reference.")
        save_reference_snapshot(current_df, reference_path, bins=bins)

        return {
            "should_retrain": False,
            "psi": None,
            "psi_threshold": psi_threshold,
            "drift_level": "no_reference",
            "current_row_count": current_row_count,
            "reference_row_count": None,
            "current_price_median": current_median,
            "reference_price_median": None,
            "reference_snapshot_date": None,
            "current_snapshot_date": current_date,
            "reason": (
                "No reference snapshot exists. Current data saved as the new reference. "
                "Retraining skipped on this first run."
            ),
        }

    ref_edges = reference.get("price_log_bin_edges")
    ref_pct = reference.get("price_log_ref_pct")
    if ref_edges and ref_pct and len(ref_edges) >= 3:
        edges_arr = np.array(ref_edges, dtype=float)
        ref_pct_arr = np.array(ref_pct, dtype=float)
        log_current = np.log1p(current_prices.values)
        log_current = np.clip(log_current, edges_arr[0], edges_arr[-1])
        cur_counts, _ = np.histogram(log_current, bins=edges_arr)
        cur_pct_arr = cur_counts / max(cur_counts.sum(), 1)

        eps = 1e-6
        ref_safe = np.where(ref_pct_arr == 0, eps, ref_pct_arr)
        cur_safe = np.where(cur_pct_arr == 0, eps, cur_pct_arr)
        psi = float(np.sum((ref_safe - cur_safe) * np.log(ref_safe / cur_safe)))
    else:
        psi = None

    if psi is None:
        logger.warning(
            "Reference snapshot is missing 'price_log_bin_edges'. "
            "Delete %s and re-run to rebuild the reference.",
            reference_path,
        )
    else:
        logger.info(
            "PSI computed using reference bin edges (ref_date=%s): %.4f",
            reference.get("snapshot_date"),
            psi,
        )

    if psi is None:
        drift_level = "unknown"
        should_retrain = False
        reason = "PSI could not be computed. Retraining skipped as a safety measure."
    elif psi >= PSI_THRESHOLD_HIGH:
        drift_level = "high"
        should_retrain = psi >= psi_threshold
        reason = (
            f"High drift (PSI={psi:.4f}). Retraining "
            f"{'triggered' if should_retrain else 'not triggered'} "
            f"(threshold={psi_threshold})."
        )
    elif psi >= PSI_THRESHOLD_MODERATE:
        drift_level = "moderate"
        should_retrain = psi >= psi_threshold
        reason = (
            f"Moderate drift (PSI={psi:.4f}). Retraining "
            f"{'triggered' if should_retrain else 'not triggered'} "
            f"(threshold={psi_threshold})."
        )
    else:
        drift_level = "none"
        should_retrain = False
        reason = f"No significant drift (PSI={psi:.4f}). Retraining skipped."

    logger.info(
        "Drift check complete | PSI=%.4f | drift_level=%s | should_retrain=%s",
        psi if psi is not None else float("nan"),
        drift_level,
        should_retrain,
    )

    if should_retrain:
        save_reference_snapshot(current_df, reference_path, bins=bins)
        logger.info("Reference snapshot updated because drift was detected.")
    else:
        logger.info("Reference snapshot not updated because no drift was detected.")

    return {
        "should_retrain": should_retrain,
        "psi": psi,
        "psi_threshold": psi_threshold,
        "drift_level": drift_level,
        "current_row_count": current_row_count,
        "reference_row_count": int(reference.get("row_count", 0)),
        "current_price_median": current_median,
        "reference_price_median": float(reference.get("price_median", 0)),
        "reference_snapshot_date": str(reference.get("snapshot_date")),
        "current_snapshot_date": current_date,
        "reason": reason,
    }


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s"
    )

    REFERENCE_PATH = DEFAULT_REFERENCE_PATH
    PSI_THRESHOLD = PSI_THRESHOLD_MODERATE

    SNAPSHOT_DATE = (date.today()).isoformat()
    logging.info("Running drift check for snapshot date: %s", SNAPSHOT_DATE)
    df = load_dashboard_df(snapshot_date=SNAPSHOT_DATE)
    result = run_drift_check(
        current_df=df,
        reference_path=REFERENCE_PATH,
        psi_threshold=PSI_THRESHOLD,
    )

    print(json.dumps(result, indent=2))
    sys.exit(0 if not result["should_retrain"] else 1)
