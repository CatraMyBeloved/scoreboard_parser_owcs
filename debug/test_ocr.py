"""Test PaddleOCR on stat crops.

Usage:
    uv run python debug/test_ocr.py [screenshot_path]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from test_utilities import (
    get_screenshot_path,
    load_screenshot_with_columns,
    print_header,
    crop_cell,
)
from src.recognize import ocr_text, _run_ocr


STAT_COLUMNS = ["elims", "assists", "deaths", "damage", "healing", "mit"]


def parse_number(text: str) -> int | None:
    """Parse a number from OCR text, handling commas."""
    if not text:
        return None
    cleaned = text.replace(',', '').replace(' ', '').strip()
    try:
        return int(cleaned)
    except ValueError:
        return None


def main() -> None:
    screenshot_path = get_screenshot_path()
    if screenshot_path is None:
        print("No screenshots found in screenshots/")
        print("Usage: uv run python debug/test_ocr.py [screenshot_path]")
        return

    print_header(f"OCR TEST: {screenshot_path.name}")

    image, columns, left_edge, right_edge = load_screenshot_with_columns(screenshot_path)
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print(f"Scoreboard width: {right_edge - left_edge}px")

    for team in [1, 2]:
        team_name = "Team 1 (Cyan)" if team == 1 else "Team 2 (Yellow)"
        print(f"\n{'=' * 50}")
        print(team_name)
        print('=' * 50)

        for row in range(5):
            name_crop = crop_cell(image, team, row, "name", columns)
            name = ocr_text(name_crop)

            stats = {}
            for column in STAT_COLUMNS:
                crop = crop_cell(image, team, row, column, columns)
                text = _run_ocr(crop)
                stats[column] = parse_number(text)

            stats_str = ', '.join(f"{k}={v}" for k, v in stats.items())
            print(f"  Row {row}: {name:15s} | {stats_str}")


if __name__ == "__main__":
    main()
