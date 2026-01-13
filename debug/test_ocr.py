"""Test PaddleOCR on stat crops.

Usage:
    uv run python debug/test_ocr.py [screenshot_path]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from paddleocr import PaddleOCR

from src import regions
from src.utils import (
    SCREENSHOTS_DIR,
    crop_cell,
    calculate_column_positions,
    detect_scoreboard_edges,
    load_image,
)

# Initialize PaddleOCR (use_angle_cls for rotated text, lang='en' for English)
ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)


def ocr_crop(crop):
    """Run OCR on a crop and return the text."""
    result = ocr.ocr(crop, cls=False)
    if result and result[0]:
        # Extract text from result
        texts = [line[1][0] for line in result[0]]
        return ' '.join(texts)
    return ''


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

    # Test OCR on various columns
    stat_columns = ["elims", "assists", "deaths", "damage", "healing", "mit"]

    print("\n=== Team 1 (Blue) ===")
    for row in range(5):
        print(f"\nRow {row}:")

        # Player name
        name_crop = crop_cell(image, 1, row, "name", columns)
        name = ocr_crop(name_crop)
        print(f"  Name: {name}")

        # Stats
        stats = []
        for col in stat_columns:
            crop = crop_cell(image, 1, row, col, columns)
            text = ocr_crop(crop)
            stats.append(f"{col}={text}")
        print(f"  Stats: {', '.join(stats)}")

    print("\n=== Team 2 (Yellow) ===")
    for row in range(5):
        print(f"\nRow {row}:")

        # Player name
        name_crop = crop_cell(image, 2, row, "name", columns)
        name = ocr_crop(name_crop)
        print(f"  Name: {name}")

        # Stats
        stats = []
        for col in stat_columns:
            crop = crop_cell(image, 2, row, col, columns)
            text = ocr_crop(crop)
            stats.append(f"{col}={text}")
        print(f"  Stats: {', '.join(stats)}")


if __name__ == "__main__":
    main()
