"""Data processing and event detection.

Handles:
- Assembling raw extractions into DataFrame
- Cleaning and validating values
- Deriving events from deltas (kills, deaths, ult usage, hero swaps)
- Export to CSV/Parquet
"""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.recognize import PlayerData, process_frame
from src.utils import load_image, OUTPUT_DIR


def build_dataframe(frames_data: list[tuple[str, list[PlayerData]]]) -> pd.DataFrame:
    """Assemble all frame data into a single DataFrame.

    Args:
        frames_data: List of (frame_id, players) tuples where players is list of PlayerData

    Returns:
        DataFrame with columns: frame, team, row, role, hero, name, ult_ready, ult_charge,
        elims, assists, deaths, damage, healing, mit
    """
    rows = []
    for frame_id, players in frames_data:
        for player in players:
            row = asdict(player)
            row["frame"] = frame_id
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Reorder columns with frame first
    cols = ["frame", "team", "row", "role", "hero", "name", "ult_ready", "ult_charge",
            "elims", "assists", "deaths", "damage", "healing", "mit"]
    df = df[cols]

    return df


def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate extracted values.

    - Strips whitespace from string columns
    - Ensures numeric columns are proper integers
    - Handles common OCR errors
    """
    if df.empty:
        return df

    df = df.copy()

    # Clean string columns
    for col in ["role", "hero", "name"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace("None", None)

    # Ensure numeric columns are integers (NaN for missing)
    numeric_cols = ["elims", "assists", "deaths", "damage", "healing", "mit", "ult_charge"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df


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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def export_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Export DataFrame to Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def process_screenshots(screenshot_paths: list[Path], output_path: Path | None = None) -> pd.DataFrame:
    """Process multiple screenshots and optionally save to CSV.

    Args:
        screenshot_paths: List of screenshot file paths
        output_path: Optional path to save CSV (defaults to output/scoreboard.csv)

    Returns:
        DataFrame with all extracted data
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "scoreboard.csv"

    frames_data = []

    for i, path in enumerate(screenshot_paths):
        frame_id = path.stem
        print(f"Processing [{i+1}/{len(screenshot_paths)}]: {path.name}")

        try:
            image = load_image(path)
            players = process_frame(image)
            frames_data.append((frame_id, players))
        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f"\nBuilding DataFrame...")
    df = build_dataframe(frames_data)
    df = clean_values(df)

    print(f"Exporting to {output_path}")
    export_csv(df, output_path)

    print(f"Done! {len(df)} rows written.")
    return df


def main():
    """Process all screenshots in the screenshots folder."""
    from src.utils import SCREENSHOTS_DIR

    # Find all PNG screenshots (exclude "full team" which are for templates)
    screenshots = sorted([
        p for p in SCREENSHOTS_DIR.glob("*.png")
        if not p.name.startswith("full team")
    ])

    if not screenshots:
        print("No screenshots found in screenshots/")
        return

    print(f"Found {len(screenshots)} screenshots")
    df = process_screenshots(screenshots[0:2])
    print(f"\nSample data:")
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()
