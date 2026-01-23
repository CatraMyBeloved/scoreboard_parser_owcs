"""Test script for examining preprocessing steps.

Crops and preprocesses a single frame, saving all intermediate steps
for visual inspection. No OCR is run.

Usage:
    python debug/test_preprocessing.py [screenshot_path]
"""

import sys
from pathlib import Path

import cv2
import numpy as np

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


# Columns to process
STAT_COLUMNS = ["name", "ult", "elims", "assists", "deaths", "damage", "healing", "mit"]
TEMPLATE_COLUMNS = ["role", "hero"]


def preprocess_for_ocr(crop: np.ndarray) -> dict[str, np.ndarray]:
    """Run preprocessing steps and return all intermediate images.

    Pipeline: upscale -> grayscale -> CLAHE -> Otsu inverted
    """
    results = {}

    # Step 1: Scale 4x
    scaled = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    results["1_scaled"] = scaled

    # Step 2: Grayscale
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    results["2_gray"] = gray

    # Step 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    results["3_clahe"] = enhanced

    # Step 4: Otsu inverted (auto threshold, white text becomes black)
    _, otsu_inv = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    results["4_otsu_inv"] = otsu_inv

    return results


def extract_saturation_binary(crop: np.ndarray) -> np.ndarray:
    """Extract binary image based on saturation (for role/hero matching)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, binary = cv2.threshold(saturation, 5, 255, cv2.THRESH_BINARY_INV)
    return binary


def process_frame(screenshot_path: Path) -> None:
    """Process a single frame and save all preprocessing steps."""
    print(f"Processing: {screenshot_path.name}")
    print("=" * 60)

    # Load image
    image = load_image(screenshot_path)
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Detect edges and calculate columns
    left_edge, right_edge = detect_scoreboard_edges(image)
    columns = calculate_column_positions(left_edge, right_edge)
    print(f"Scoreboard edges: left={left_edge}, right={right_edge}")

    # Create output directory for this frame
    frame_name = screenshot_path.stem.replace(" ", "_")
    frame_dir = OUTPUT_DIR / "test" / frame_name
    frame_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {frame_dir}")
    print()

    for team in (1, 2):
        team_dir = frame_dir / f"team{team}"
        team_dir.mkdir(exist_ok=True)
        print(f"Team {team}:")

        for row in range(5):
            row_prefix = f"p{row + 1}"  # p1, p2, p3, p4, p5

            # Process stat columns (OCR preprocessing)
            for col_name in STAT_COLUMNS:
                if col_name not in columns:
                    continue

                crop = crop_cell(image, team, row, col_name, columns)

                # Save original
                save_crop(crop, team_dir / f"{row_prefix}_{col_name}_0_original.png")

                # Save preprocessing steps
                steps = preprocess_for_ocr(crop)
                for step_name, step_img in steps.items():
                    save_crop(step_img, team_dir / f"{row_prefix}_{col_name}_{step_name}.png")

            # Process template columns (role, hero)
            for col_name in TEMPLATE_COLUMNS:
                if col_name not in columns:
                    continue

                crop = crop_cell(image, team, row, col_name, columns)

                # Save original
                save_crop(crop, team_dir / f"{row_prefix}_{col_name}_0_original.png")

                # Save saturation binary (used for template matching)
                sat_binary = extract_saturation_binary(crop)
                save_crop(sat_binary, team_dir / f"{row_prefix}_{col_name}_1_sat_binary.png")

            print(f"  Player {row + 1}: saved")

    print()
    print(f"Done! Output saved to: {frame_dir}")


def main():
    """Run preprocessing on a single screenshot."""
    if len(sys.argv) > 1:
        screenshot_path = Path(sys.argv[1])
        if not screenshot_path.is_absolute():
            screenshot_path = SCREENSHOTS_DIR / screenshot_path
    else:
        # Find first PNG in screenshots folder
        screenshots = sorted(SCREENSHOTS_DIR.glob("*.png"))

        if not screenshots:
            print("No screenshots found in screenshots/")
            print("Usage: python debug/test_preprocessing.py [screenshot_path]")
            return

        screenshot_path = screenshots[0]

    if not screenshot_path.exists():
        print(f"Screenshot not found: {screenshot_path}")
        return

    process_frame(screenshot_path)


if __name__ == "__main__":
    main()
