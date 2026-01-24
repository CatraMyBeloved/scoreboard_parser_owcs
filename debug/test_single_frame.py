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

sys.path.insert(0, str(Path(__file__).parent.parent))

from test_utilities import (
    get_screenshot_path,
    load_screenshot_with_columns,
    print_header,
    print_section,
    crop_cell,
    save_crop,
    OUTPUT_DIR,
)
from src.recognize import extract_by_saturation, process_frame
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
                    continue

                crop = crop_cell(image, team, row, col_name, columns)
                filename = f"row{row}_{col_name}.png"
                save_crop(crop, team_dir / filename)


def save_preprocessed_crops(image: np.ndarray, output_dir: Path, columns: dict) -> None:
    """Save preprocessed versions of crops for debugging."""
    output_dir.mkdir(parents=True, exist_ok=True)

    stat_columns = ["elims", "assists", "deaths", "damage", "healing", "mit"]
    ocr_columns = ["name", "ult"]
    template_columns = ["role", "hero"]

    for team in (1, 2):
        team_dir = output_dir / f"team{team}"
        team_dir.mkdir(exist_ok=True)

        for row in range(5):
            for col_name in stat_columns:
                if col_name not in columns:
                    continue

                crop = crop_cell(image, team, row, col_name, columns)
                save_crop(crop, team_dir / f"row{row}_{col_name}_1_original.png")

                scaled = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
                save_crop(scaled, team_dir / f"row{row}_{col_name}_2_scaled.png")

                gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
                save_crop(gray, team_dir / f"row{row}_{col_name}_3_gray.png")

                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                save_crop(enhanced, team_dir / f"row{row}_{col_name}_4_clahe.png")

                _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                save_crop(otsu, team_dir / f"row{row}_{col_name}_5_otsu.png")

            for col_name in ocr_columns:
                if col_name not in columns:
                    continue

                crop = crop_cell(image, team, row, col_name, columns)
                save_crop(crop, team_dir / f"row{row}_{col_name}_1_original.png")

                scaled = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                save_crop(scaled, team_dir / f"row{row}_{col_name}_2_scaled.png")

                hsv = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)
                saturation = hsv[:, :, 1]
                _, sat_binary = cv2.threshold(saturation, 30, 255, cv2.THRESH_BINARY)
                save_crop(sat_binary, team_dir / f"row{row}_{col_name}_3_saturation.png")

            for col_name in template_columns:
                if col_name not in columns:
                    continue

                crop = crop_cell(image, team, row, col_name, columns)
                save_crop(crop, team_dir / f"row{row}_{col_name}_1_original.png")

                sat_binary = extract_by_saturation(crop)
                save_crop(sat_binary, team_dir / f"row{row}_{col_name}_2_saturation_binary.png")


def process_single_frame(screenshot_path: Path) -> None:
    """Process a single frame and output all debug info."""
    print_header(f"SINGLE FRAME TEST: {screenshot_path.name}")

    image, columns, left_edge, right_edge = load_screenshot_with_columns(screenshot_path)
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print(f"Scoreboard edges: left={left_edge}, right={right_edge}")
    print(f"Scoreboard width: {right_edge - left_edge}px\n")

    test_output = OUTPUT_DIR / "test"
    crops_dir = test_output / "crops"
    preprocessed_dir = test_output / "preprocessed"

    print("Saving crops...")
    save_all_crops(image, crops_dir, columns)
    print(f"  Crops saved to: {crops_dir}")

    print("Saving preprocessed images...")
    save_preprocessed_crops(image, preprocessed_dir, columns)
    print(f"  Preprocessed saved to: {preprocessed_dir}")

    print("\nRunning recognition...")
    print("  - Stats: digit template matching")
    print("  - Names/Ult: PaddleOCR")
    players = process_frame(image)

    frame_id = screenshot_path.stem
    dataframe = build_dataframe([(frame_id, players)])
    dataframe = clean_values(dataframe)

    csv_path = test_output / "output.csv"
    export_csv(dataframe, csv_path)
    print(f"  CSV saved to: {csv_path}")

    print_header("EXTRACTED DATA")

    for team in (1, 2):
        team_name = "Team 1 (Cyan)" if team == 1 else "Team 2 (Yellow)"
        print_section(team_name)

        team_dataframe = dataframe[dataframe["team"] == team]
        for _, row in team_dataframe.iterrows():
            role = row['role'] or '?'
            hero = row['hero'] or '?'
            name = row['name'] or '?'
            print(f"  [{role:>7}] {hero:<15} {name:<20}")
            ult_str = 'READY' if row['ult_ready'] else f"{row['ult_charge']}%" if row['ult_charge'] is not None else '?'
            print(f"           Ult: {ult_str}")
            print(f"           E:{row['elims']} A:{row['assists']} D:{row['deaths']} "
                  f"DMG:{row['damage']} HEAL:{row['healing']} MIT:{row['mit']}")
            print()

    print_header("Output files")
    print(f"  - Crops:        {crops_dir}")
    print(f"  - Preprocessed: {preprocessed_dir}")
    print(f"  - CSV:          {csv_path}")


def main() -> None:
    screenshot_path = get_screenshot_path()
    if screenshot_path is None:
        print("No screenshots found in screenshots/")
        print("Usage: uv run python debug/test_single_frame.py [screenshot_path]")
        return

    process_single_frame(screenshot_path)


if __name__ == "__main__":
    main()
