"""Compare different preprocessing approaches on a single crop.

Usage:
    python debug/compare_preprocessing.py
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


def run_all_methods(crop: np.ndarray) -> dict[str, np.ndarray]:
    """Run all preprocessing methods and return results."""
    results = {}

    # Scale 4x for all methods
    scaled = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    results["00_original"] = crop
    results["01_scaled"] = scaled

    # --- Method 1: Raw grayscale (let OCR handle it) ---
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    results["02_gray_raw"] = gray

    # --- Method 2: Otsu's automatic threshold ---
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results["03_otsu"] = otsu

    # --- Method 3: Otsu inverted ---
    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    results["04_otsu_inv"] = otsu_inv

    # --- Method 4: Adaptive threshold ---
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    results["05_adaptive"] = adaptive

    # --- Method 5: Adaptive threshold inverted ---
    adaptive_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)
    results["06_adaptive_inv"] = adaptive_inv

    # --- Method 6: Value channel (brightness) ---
    hsv = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    results["07_value_raw"] = value

    # --- Method 7: Value + Otsu ---
    _, value_otsu = cv2.threshold(value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results["08_value_otsu"] = value_otsu

    # --- Method 8: Saturation channel ---
    saturation = hsv[:, :, 1]
    results["09_saturation_raw"] = saturation

    # --- Method 9: Saturation + Otsu ---
    _, sat_otsu = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results["10_saturation_otsu"] = sat_otsu

    # --- Method 10: Saturation inverted (low sat = white) ---
    _, sat_otsu_inv = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    results["11_saturation_otsu_inv"] = sat_otsu_inv

    # --- Method 11: Color mask - remove cyan ---
    hsv_full = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)
    # Cyan is roughly H=80-100 in OpenCV (0-180 range)
    lower_cyan = np.array([80, 50, 50])
    upper_cyan = np.array([100, 255, 255])
    cyan_mask = cv2.inRange(hsv_full, lower_cyan, upper_cyan)
    no_cyan = cv2.bitwise_not(cyan_mask)
    results["12_no_cyan_mask"] = no_cyan

    # --- Method 12: Color mask - remove yellow ---
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv_full, lower_yellow, upper_yellow)
    no_yellow = cv2.bitwise_not(yellow_mask)
    results["13_no_yellow_mask"] = no_yellow

    # --- Method 13: Combined - remove both cyan and yellow ---
    combined_mask = cv2.bitwise_or(cyan_mask, yellow_mask)
    no_colors = cv2.bitwise_not(combined_mask)
    results["14_no_cyan_yellow"] = no_colors

    # --- Method 14: High value + low saturation (white detection) ---
    white_mask = cv2.bitwise_and(
        cv2.inRange(value, 200, 255),  # high brightness
        cv2.inRange(saturation, 0, 30)  # low saturation
    )
    results["15_white_mask"] = white_mask

    # --- Method 15: Saturation threshold at 15 (our current) ---
    _, sat_15 = cv2.threshold(saturation, 15, 255, cv2.THRESH_BINARY)
    results["16_sat_thresh_15"] = sat_15

    # --- Method 16: Saturation threshold at 15, inverted ---
    _, sat_15_inv = cv2.threshold(saturation, 15, 255, cv2.THRESH_BINARY_INV)
    results["17_sat_thresh_15_inv"] = sat_15_inv

    return results


def main():
    # Find first screenshot
    screenshots = sorted(SCREENSHOTS_DIR.glob("*.png"))
    if not screenshots:
        print("No screenshots found")
        return

    screenshot_path = screenshots[0]
    print(f"Using: {screenshot_path.name}")

    # Load and detect
    image = load_image(screenshot_path)
    left_edge, right_edge = detect_scoreboard_edges(image)
    columns = calculate_column_positions(left_edge, right_edge)

    # Get p1 damage crop (team 1, row 0, damage column)
    crop = crop_cell(image, 1, 0, "damage", columns)

    # Run all methods
    results = run_all_methods(crop)

    # Save all results
    output_dir = OUTPUT_DIR / "test" / "compare_methods"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, img in results.items():
        save_crop(img, output_dir / f"{name}.png")
        print(f"Saved: {name}.png")

    print(f"\nAll outputs saved to: {output_dir}")
    print("\nMethods:")
    print("  02-06: Grayscale-based (raw, otsu, adaptive)")
    print("  07-08: Value channel (brightness)")
    print("  09-11: Saturation channel")
    print("  12-14: Color masking (remove cyan/yellow)")
    print("  15-17: Combined/threshold approaches")


if __name__ == "__main__":
    main()
