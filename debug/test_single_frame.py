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
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    extract_by_saturation,
    process_frame,
    PlayerData,
)
from src.digit_matcher import get_digit_matcher
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
    """Save preprocessed versions of crops for debugging."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stat columns use digit template matching
    stat_columns = ["elims", "assists", "deaths", "damage", "healing", "mit"]

    # OCR columns (name, ult) use PaddleOCR with saturation preprocessing
    ocr_columns = ["name", "ult"]

    # Template matching columns
    template_columns = ["role", "hero"]

    for team in (1, 2):
        team_dir = output_dir / f"team{team}"
        team_dir.mkdir(exist_ok=True)

        for row in range(5):
            # Process stat columns (digit matching preprocessing)
            for col_name in stat_columns:
                if col_name not in columns:
                    continue

                crop = crop_cell(image, team, row, col_name, columns)

                # Save original crop
                orig_path = team_dir / f"row{row}_{col_name}_1_original.png"
                save_crop(crop, orig_path)

                # Save scaled version (4x) - digit matcher uses this
                scaled = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
                scaled_path = team_dir / f"row{row}_{col_name}_2_scaled.png"
                save_crop(scaled, scaled_path)

                # Save grayscale
                gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
                gray_path = team_dir / f"row{row}_{col_name}_3_gray.png"
                save_crop(gray, gray_path)

                # CLAHE enhanced
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                clahe_path = team_dir / f"row{row}_{col_name}_4_clahe.png"
                save_crop(enhanced, clahe_path)

                # Otsu (not inverted - digit matcher style)
                _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                otsu_path = team_dir / f"row{row}_{col_name}_5_otsu.png"
                save_crop(otsu, otsu_path)

            # Process OCR columns (name, ult)
            for col_name in ocr_columns:
                if col_name not in columns:
                    continue

                crop = crop_cell(image, team, row, col_name, columns)

                # Save original
                orig_path = team_dir / f"row{row}_{col_name}_1_original.png"
                save_crop(crop, orig_path)

                # Scaled (3x for OCR)
                scaled = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                scaled_path = team_dir / f"row{row}_{col_name}_2_scaled.png"
                save_crop(scaled, scaled_path)

                # Saturation extraction (for white text on colored bg)
                hsv = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)
                saturation = hsv[:, :, 1]
                _, sat_binary = cv2.threshold(saturation, 30, 255, cv2.THRESH_BINARY)
                sat_path = team_dir / f"row{row}_{col_name}_3_saturation.png"
                save_crop(sat_binary, sat_path)

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

    # Process frame with recognition
    print()
    print("Running recognition...")
    print("  - Stats: digit template matching")
    print("  - Names/Ult: PaddleOCR")
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
            role = row['role'] or '?'
            hero = row['hero'] or '?'
            name = row['name'] or '?'
            print(f"  [{role:>7}] {hero:<15} {name:<20}")
            ult_str = 'READY' if row['ult_ready'] else f"{row['ult_charge']}%" if row['ult_charge'] is not None else '?'
            print(f"           Ult: {ult_str}")
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
        # Find first PNG in screenshots folder
        screenshots = sorted(SCREENSHOTS_DIR.glob("*.png"))

        if not screenshots:
            print("No screenshots found in screenshots/")
            print("Usage: python debug/test_single_frame.py [screenshot_path]")
            return

        screenshot_path = screenshots[0]

    if not screenshot_path.exists():
        print(f"Screenshot not found: {screenshot_path}")
        return

    process_single_frame(screenshot_path)


if __name__ == "__main__":
    main()
