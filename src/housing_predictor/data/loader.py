"""Data loading utilities."""

from pathlib import Path

import pandas as pd


def load_dataframe(data_path: str) -> pd.DataFrame:
    """Load a CSV or ZIP (first CSV file) into a DataFrame."""
    path = Path(data_path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".zip":
        import zipfile

        extract_dir = Path("data/extracted")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        csv_files = sorted(extract_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in zip archive: {path}")
        return pd.read_csv(csv_files[0])
    raise ValueError(f"Unsupported format: {path.suffix}")
