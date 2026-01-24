"""Test script for examining preprocessing steps.

Crops and preprocesses a single frame, saving all intermediate steps
for visual inspection. No OCR is run.

Usage:
    uv run python debug/test_preprocessing.py [screenshot_path]
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from test_utilities import (
    get_screenshot_path,
    load_screenshot_with_columns,
    create_output_dir,
    print_header,
    crop_cell,
    save_crop,
)
from src.recognize import extract_by_saturation


STAT_COLUMNS = ["name", "ult", "elims", "assists", "deaths", "damage", "healing", "mit"]
TEMPLATE_COLUMNS = ["role", "hero"]


def preprocess_for_ocr(crop: np.ndarray) -> dict[str, np.ndarray]:
    """Run preprocessing steps and return all intermediate images.

    Pipeline: upscale -> grayscale -> CLAHE -> Otsu inverted
    """
    results = {}

    scaled = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    results["1_scaled"] = scaled

    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    results["2_gray"] = gray

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    results["3_clahe"] = enhanced

    _, otsu_inv = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    results["4_otsu_inv"] = otsu_inv

    return results


def main() -> None:
    screenshot_path = get_screenshot_path()
    if screenshot_path is None:
        print("No screenshots found in screenshots/")
        print("Usage: uv run python debug/test_preprocessing.py [screenshot_path]")
        return

    print_header(f"PREPROCESSING TEST: {screenshot_path.name}")

    image, columns, left_edge, right_edge = load_screenshot_with_columns(screenshot_path)
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print(f"Scoreboard edges: left={left_edge}, right={right_edge}")

    output_dir = create_output_dir(screenshot_path.stem)
    print(f"Output: {output_dir}\n")

    for team in (1, 2):
        team_dir = output_dir / f"team{team}"
        team_dir.mkdir(exist_ok=True)
        print(f"Team {team}:")

        for row in range(5):
            row_prefix = f"p{row + 1}"

            for col_name in STAT_COLUMNS:
                if col_name not in columns:
                    continue

                crop = crop_cell(image, team, row, col_name, columns)
                save_crop(crop, team_dir / f"{row_prefix}_{col_name}_0_original.png")

                steps = preprocess_for_ocr(crop)
                for step_name, step_image in steps.items():
                    save_crop(step_image, team_dir / f"{row_prefix}_{col_name}_{step_name}.png")

            for col_name in TEMPLATE_COLUMNS:
                if col_name not in columns:
                    continue

                crop = crop_cell(image, team, row, col_name, columns)
                save_crop(crop, team_dir / f"{row_prefix}_{col_name}_0_original.png")

                sat_binary = extract_by_saturation(crop, threshold=5)
                save_crop(sat_binary, team_dir / f"{row_prefix}_{col_name}_1_sat_binary.png")

            print(f"  Player {row + 1}: saved")

    print(f"\nDone! Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
