"""Debug script to visualize detected column boundaries.

Usage:
    uv run python debug/show_columns.py [screenshot_path]
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from test_utilities import (
    get_screenshot_path,
    load_screenshot_with_columns,
    OUTPUT_DIR,
    regions,
)
from src.utils import ALL_COLUMNS


def draw_column_boundaries(
    image: np.ndarray,
    columns: dict[str, tuple[int, int]],
    left_edge: int,
    right_edge: int,
    team_rows: list[int],
    team_color: tuple[int, int, int],
) -> np.ndarray:
    """Draw column boundaries for one team's scoreboard."""
    result = image.copy()

    y_start = team_rows[0]
    y_end = team_rows[-1] + regions.ROW_HEIGHT

    for col_name, (x_start, x_end) in columns.items():
        cv2.line(result, (x_start, y_start), (x_start, y_end), team_color, 1)
        cv2.line(result, (x_end, y_start), (x_end, y_end), team_color, 1)

        label_y = y_start - 5
        cv2.putText(result, col_name, (x_start + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, team_color, 1)

    for row_y in team_rows:
        cv2.line(result, (left_edge, row_y), (right_edge, row_y), team_color, 1)
    cv2.line(result, (left_edge, y_end), (right_edge, y_end), team_color, 1)

    return result


def main() -> None:
    screenshot_path = get_screenshot_path()
    if screenshot_path is None:
        print("No screenshots found in screenshots/")
        print("Usage: uv run python debug/show_columns.py [screenshot_path]")
        sys.exit(1)

    print(f"Loading {screenshot_path}...")
    image, columns, left_edge, right_edge = load_screenshot_with_columns(screenshot_path)
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print(f"Detected edges: left={left_edge}, right={right_edge}")
    print(f"Scoreboard width: {right_edge - left_edge}px")

    print("\nColumn positions:")
    for col_name in ALL_COLUMNS:
        x_start, x_end = columns[col_name]
        print(f"  {col_name}: {x_start}-{x_end} (width={x_end - x_start})")

    result = image.copy()
    result = draw_column_boundaries(
        result, columns, left_edge, right_edge, regions.TEAM1_ROWS, (255, 200, 100)
    )
    result = draw_column_boundaries(
        result, columns, left_edge, right_edge, regions.TEAM2_ROWS, (100, 200, 255)
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "debug_columns.png"
    cv2.imwrite(str(output_path), result)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
