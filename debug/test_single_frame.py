"""Test script for processing a single frame.

Outputs:
- All cell crops to output/test_crops/
- Preprocessed versions of each crop to output/test_preprocessed/
- CSV with extracted data to output/test_output.csv
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import (
    load_image,
    save_crop,
    detect_scoreboard_edges,
    calculate_column_positions,
    crop_cell,
    SCREENSHOTS_DIR,
    OUTPUT_DIR,
)
from src import regions
from src.recognize import (
    _preprocess_for_ocr,
    extract_by_saturation,
    process_frame,
    PlayerData,
)
from src.process import build_dataframe, clean_values, export_csv


def save_all_crops(image: np.ndarray, output_dir: Path, columns: dict) -> None:
    """Save all cell crops from a frame."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for team in (1, 2):
        team_dir = output_dir / f"team{team}"
        team_dir.mkdir(exist_ok=True)

        for row in range(5):
            for col_name in columns.keys():
                if col_name == "perks":
                    continue  # Skip perks column

                crop = crop_cell(image, team, row, col_name, columns)
                filename = f"row{row}_{col_name}.png"
                save_crop(crop, team_dir / filename)


def save_preprocessed_crops(image: np.ndarray, output_dir: Path, columns: dict) -> None:
    """Save preprocessed versions of crops for debugging OCR."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Columns that use OCR (saturation-based preprocessing)
    ocr_columns = ["name", "ult", "elims", "assists", "deaths", "damage", "healing", "mit"]

    # Columns that use saturation extraction (template matching)
    template_columns = ["role", "hero"]

    for team in (1, 2):
        team_dir = output_dir / f"team{team}"
        team_dir.mkdir(exist_ok=True)

        for row in range(5):
            # Process OCR columns
            for col_name in ocr_columns:
                if col_name not in columns:
                    continue

                crop = crop_cell(image, team, row, col_name, columns)

                # Save original crop
                orig_path = team_dir / f"row{row}_{col_name}_1_original.png"
                save_crop(crop, orig_path)

                # Save scaled version (3x)
                scaled = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                scaled_path = team_dir / f"row{row}_{col_name}_2_scaled.png"
                save_crop(scaled, scaled_path)

                # Save saturation channel
                hsv = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)
                saturation = hsv[:, :, 1]
                sat_path = team_dir / f"row{row}_{col_name}_3_saturation.png"
                save_crop(saturation, sat_path)

                # Save binary threshold
                _, binary = cv2.threshold(saturation, 30, 255, cv2.THRESH_BINARY)
                binary_path = team_dir / f"row{row}_{col_name}_4_binary.png"
                save_crop(binary, binary_path)

                # Save eroded (final OCR input)
                kernel = np.ones((2, 2), np.uint8)
                eroded = cv2.erode(binary, kernel, iterations=1)
                eroded_path = team_dir / f"row{row}_{col_name}_5_eroded.png"
                save_crop(eroded, eroded_path)

                # Save full preprocessing result
                preprocessed = _preprocess_for_ocr(crop)
                final_path = team_dir / f"row{row}_{col_name}_6_final.png"
                save_crop(preprocessed, final_path)

            # Process template matching columns (role, hero)
            for col_name in template_columns:
                if col_name not in columns:
                    continue

                crop = crop_cell(image, team, row, col_name, columns)

                # Save original
                orig_path = team_dir / f"row{row}_{col_name}_1_original.png"
                save_crop(crop, orig_path)

                # Save saturation extraction (used for role matching)
                sat_binary = extract_by_saturation(crop)
                sat_path = team_dir / f"row{row}_{col_name}_2_saturation_binary.png"
                save_crop(sat_binary, sat_path)


def process_single_frame(screenshot_path: Path) -> None:
    """Process a single frame and output all debug info."""
    print(f"Processing: {screenshot_path}")
    print("=" * 60)

    # Load image
    image = load_image(screenshot_path)
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Detect edges and calculate columns
    left_edge, right_edge = detect_scoreboard_edges(image)
    columns = calculate_column_positions(left_edge, right_edge)
    print(f"Scoreboard edges: left={left_edge}, right={right_edge}")
    print(f"Scoreboard width: {right_edge - left_edge}px")
    print()

    # Create output directories
    test_output = OUTPUT_DIR / "test"
    crops_dir = test_output / "crops"
    preprocessed_dir = test_output / "preprocessed"

    # Save all crops
    print("Saving crops...")
    save_all_crops(image, crops_dir, columns)
    print(f"  Crops saved to: {crops_dir}")

    # Save preprocessed versions
    print("Saving preprocessed images...")
    save_preprocessed_crops(image, preprocessed_dir, columns)
    print(f"  Preprocessed saved to: {preprocessed_dir}")

    # Process frame with OCR
    print()
    print("Running recognition (this may take a moment)...")
    players = process_frame(image)

    # Build and clean DataFrame
    frame_id = screenshot_path.stem
    df = build_dataframe([(frame_id, players)])
    df = clean_values(df)

    # Export CSV
    csv_path = test_output / "output.csv"
    export_csv(df, csv_path)
    print(f"  CSV saved to: {csv_path}")

    # Print results
    print()
    print("=" * 60)
    print("EXTRACTED DATA:")
    print("=" * 60)
    print()

    for team in (1, 2):
        team_name = "Team 1 (Cyan)" if team == 1 else "Team 2 (Yellow)"
        print(f"{team_name}:")
        print("-" * 40)

        team_df = df[df["team"] == team]
        for _, row in team_df.iterrows():
            print(f"  [{row['role']:>7}] {row['hero']:<15} {row['name']:<20}")
            print(f"           Ult: {'READY' if row['ult_ready'] else f'{row["ult_charge"]}%' if row['ult_charge'] is not None else '?'}")
            print(f"           E:{row['elims']} A:{row['assists']} D:{row['deaths']} "
                  f"DMG:{row['damage']} HEAL:{row['healing']} MIT:{row['mit']}")
            print()

    print("=" * 60)
    print("Output files:")
    print(f"  - Crops:        {crops_dir}")
    print(f"  - Preprocessed: {preprocessed_dir}")
    print(f"  - CSV:          {csv_path}")


def main():
    """Run test on a single screenshot."""
    # Default to first available screenshot
    if len(sys.argv) > 1:
        screenshot_path = Path(sys.argv[1])
        if not screenshot_path.is_absolute():
            screenshot_path = SCREENSHOTS_DIR / screenshot_path
    else:
        # Find first "full team" screenshot for testing
        screenshots = sorted(SCREENSHOTS_DIR.glob("full team *.png"))
        if not screenshots:
            # Fall back to any PNG
            screenshots = sorted(SCREENSHOTS_DIR.glob("*.png"))

        if not screenshots:
            print("No screenshots found in screenshots/")
            print("Usage: python test_single_frame.py [screenshot_path]")
            return

        screenshot_path = screenshots[0]

    if not screenshot_path.exists():
        print(f"Screenshot not found: {screenshot_path}")
        return

    process_single_frame(screenshot_path)


if __name__ == "__main__":
    main()
