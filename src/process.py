"""Data processing and event detection.

Handles:
- Assembling raw extractions into DataFrame
- Cleaning and validating values
- Deriving events from deltas (kills, deaths, ult usage, hero swaps)
- Export to CSV/Parquet
"""

from pathlib import Path

import pandas as pd


def build_dataframe(frames_data: list[dict]) -> pd.DataFrame:
    """Assemble all frame data into a single DataFrame."""
    # TODO: Implement
    pass


def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate extracted values."""
    # TODO: Strip commas, handle OCR errors
    pass


def detect_events(df: pd.DataFrame) -> pd.DataFrame:
    """Derive events from stat deltas.

    Detectable events:
    - Hero switches (portrait changes)
    - Deaths (death count increments)
    - Kills (elim count increments)
    - Ult gained (charge hits 100% / ready)
    - Ult used (ready -> 0%)
    """
    # TODO: Implement event detection
    pass


def export_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Export DataFrame to CSV."""
    df.to_csv(output_path, index=False)


def export_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Export DataFrame to Parquet."""
    df.to_parquet(output_path, index=False)
