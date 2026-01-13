"""Test PaddleOCR on stat crops.

Usage:
    uv run python debug/test_ocr.py [screenshot_path]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import regions
from src.utils import (
    SCREENSHOTS_DIR,
    crop_cell,
    calculate_column_positions,
    detect_scoreboard_edges,
    load_image,
)

# Lazy-loaded OCR instance
_ocr = None


def get_ocr():
    """Get or initialize PaddleOCR instance."""
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
        print("Initializing PaddleOCR...")
        _ocr = PaddleOCR(lang='en')
    return _ocr


def ocr_crop(crop):
    """Run OCR on a crop and return the raw text."""
    ocr = get_ocr()
    result = ocr.predict(crop)
    # Result is a list of dicts, first element contains 'rec_texts'
    if result and len(result) > 0:
        first = result[0]
        if isinstance(first, dict) and 'rec_texts' in first:
            texts = first['rec_texts']
            if texts:
                return ' '.join(texts)
    return ''


def parse_number(text: str) -> int | None:
    """Parse a number from OCR text, handling commas."""
    if not text:
        return None
    # Remove commas and whitespace
    cleaned = text.replace(',', '').replace(' ', '').strip()
    # Try to parse as integer
    try:
        return int(cleaned)
    except ValueError:
        return None


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
    print(f"Scoreboard width: {right_edge - left_edge}px")

    # Test OCR on various columns
    stat_columns = ["elims", "assists", "deaths", "damage", "healing", "mit"]

    for team in [1, 2]:
        team_name = "Team 1 (Blue)" if team == 1 else "Team 2 (Yellow)"
        print(f"\n{'='*50}")
        print(f"{team_name}")
        print('='*50)

        for row in range(5):
            # Player name
            name_crop = crop_cell(image, team, row, "name", columns)
            name = ocr_crop(name_crop)

            # Stats
            stats = {}
            for col in stat_columns:
                crop = crop_cell(image, team, row, col, columns)
                text = ocr_crop(crop)
                stats[col] = parse_number(text)

            # Format output
            stats_str = ', '.join(f"{k}={v}" for k, v in stats.items())
            print(f"  Row {row}: {name:15s} | {stats_str}")


if __name__ == "__main__":
    main()
