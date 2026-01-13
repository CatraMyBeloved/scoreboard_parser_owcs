"""Debug script to visualize OCR preprocessing.

Usage:
    uv run python debug/test_preprocessing.py [screenshot_path]
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    OUTPUT_DIR,
    SCREENSHOTS_DIR,
    calculate_column_positions,
    crop_cell,
    detect_scoreboard_edges,
    load_image,
)


def preprocess_saturation(crop: np.ndarray, threshold: int = 50) -> np.ndarray:
    """Extract white text using saturation threshold."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, binary = cv2.threshold(saturation, threshold, 255, cv2.THRESH_BINARY)
    return binary


def main() -> None:
    # Find screenshot to process
    if len(sys.argv) > 1:
        screenshot_path = Path(sys.argv[1])
    else:
        images = list(SCREENSHOTS_DIR.glob("*.png")) + list(SCREENSHOTS_DIR.glob("*.jpg"))
        if not images:
            print(f"No images found in {SCREENSHOTS_DIR}")
            sys.exit(1)
        screenshot_path = sorted(images)[0]
        print(f"No image specified, using: {screenshot_path}")

    # Load image
    print(f"Loading {screenshot_path}...")
    image = load_image(screenshot_path)

    # Detect edges and calculate columns
    left_edge, right_edge = detect_scoreboard_edges(image)
    columns = calculate_column_positions(left_edge, right_edge)

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Test different thresholds
    thresholds = [30, 50, 70, 90]

    # Collect sample crops for visualization
    samples = [
        (1, 0, "name", "T1_R0_name"),
        (1, 4, "name", "T1_R4_name"),  # VOLTSA
        (2, 0, "name", "T2_R0_name"),  # COWBOY
        (1, 0, "elims", "T1_R0_elims"),
        (2, 2, "elims", "T2_R2_elims"),  # MAPPSY
        (2, 4, "deaths", "T2_R4_deaths"),  # PETITPIGEON
    ]

    for team, row, col, label in samples:
        crop = crop_cell(image, team, row, col, columns)

        # Create comparison strip
        h, w = crop.shape[:2]
        strip_h = h
        strip_w = w * (len(thresholds) + 1) + 10 * len(thresholds)
        strip = np.ones((strip_h, strip_w, 3), dtype=np.uint8) * 128

        # Original
        strip[:h, :w] = crop

        # Different thresholds
        x_offset = w + 10
        for thresh in thresholds:
            binary = preprocess_saturation(crop, thresh)
            binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            strip[:h, x_offset:x_offset + w] = binary_bgr
            x_offset += w + 10

        # Save strip
        output_path = OUTPUT_DIR / f"preprocess_{label}.png"
        cv2.imwrite(str(output_path), strip)
        print(f"Saved {output_path}")

    # Also save a combined visualization
    print(f"\nSaved preprocessing comparisons to {OUTPUT_DIR}")
    print("Columns: Original, thresh=30, thresh=50, thresh=70, thresh=90")


if __name__ == "__main__":
    main()
