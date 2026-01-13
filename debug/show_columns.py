"""Debug script to visualize detected column boundaries.

Usage:
    uv run python debug/show_columns.py [screenshot_path]
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import regions
from src.utils import (
    ALL_COLUMNS,
    OUTPUT_DIR,
    SCREENSHOTS_DIR,
    calculate_column_positions,
    detect_scoreboard_edges,
    load_image,
)


def draw_column_boundaries(
    image: np.ndarray,
    left_edge: int,
    right_edge: int,
    team_rows: list[int],
    team_color: tuple[int, int, int],
) -> np.ndarray:
    """Draw column boundaries for one team's scoreboard."""
    result = image.copy()

    columns = calculate_column_positions(left_edge, right_edge)

    y_start = team_rows[0]
    y_end = team_rows[-1] + regions.ROW_HEIGHT

    # Draw each column boundary
    for col_name, (x_start, x_end) in columns.items():
        # Vertical lines at column boundaries
        cv2.line(result, (x_start, y_start), (x_start, y_end), team_color, 1)
        cv2.line(result, (x_end, y_start), (x_end, y_end), team_color, 1)

        # Label at top
        label_y = y_start - 5
        cv2.putText(
            result,
            col_name,
            (x_start + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            team_color,
            1,
        )

    # Draw row boundaries
    for row_y in team_rows:
        cv2.line(result, (left_edge, row_y), (right_edge, row_y), team_color, 1)
    # Bottom edge
    cv2.line(result, (left_edge, y_end), (right_edge, y_end), team_color, 1)

    return result


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

    if not screenshot_path.exists():
        print(f"File not found: {screenshot_path}")
        sys.exit(1)

    # Load image
    print(f"Loading {screenshot_path}...")
    image = load_image(screenshot_path)
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Detect edges
    try:
        left_edge, right_edge = detect_scoreboard_edges(image)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Detected edges: left={left_edge}, right={right_edge}")
    print(f"Scoreboard width: {right_edge - left_edge}px")

    # Calculate and print column positions
    columns = calculate_column_positions(left_edge, right_edge)
    print("\nColumn positions:")
    for col_name in ALL_COLUMNS:
        x_start, x_end = columns[col_name]
        print(f"  {col_name}: {x_start}-{x_end} (width={x_end - x_start})")

    # Draw boundaries
    result = image.copy()
    result = draw_column_boundaries(
        result, left_edge, right_edge, regions.TEAM1_ROWS, (255, 200, 100)
    )
    result = draw_column_boundaries(
        result, left_edge, right_edge, regions.TEAM2_ROWS, (100, 200, 255)
    )

    # Save result
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "debug_columns.png"
    cv2.imwrite(str(output_path), result)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
