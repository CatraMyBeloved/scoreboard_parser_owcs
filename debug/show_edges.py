"""Debug script to visualize detected scoreboard edges.

Usage:
    uv run python debug/show_edges.py [screenshot_path]
"""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from test_utilities import (
    get_screenshot_path,
    load_image,
    detect_scoreboard_edges,
    OUTPUT_DIR,
    regions,
)


def main() -> None:
    screenshot_path = get_screenshot_path()
    if screenshot_path is None:
        print("No screenshots found in screenshots/")
        print("Usage: uv run python debug/show_edges.py [screenshot_path]")
        sys.exit(1)

    print(f"Loading {screenshot_path}...")
    image = load_image(screenshot_path)
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    print("Detecting scoreboard edges...")
    try:
        left_edge, right_edge = detect_scoreboard_edges(image)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Detected edges: left={left_edge}, right={right_edge}")
    print(f"Scoreboard width: {right_edge - left_edge}px")

    result = image.copy()
    scan_y = regions.TEAM1_ROWS[0] - 5

    cv2.line(result, (0, scan_y), (image.shape[1], scan_y), (0, 255, 255), 1)
    cv2.circle(result, (left_edge, scan_y), 5, (0, 255, 0), -1)
    cv2.circle(result, (right_edge, scan_y), 5, (0, 0, 255), -1)

    team1_y_start = regions.TEAM1_ROWS[0]
    team1_y_end = regions.TEAM1_ROWS[-1] + regions.ROW_HEIGHT
    cv2.rectangle(result, (left_edge, team1_y_start), (right_edge, team1_y_end), (255, 100, 100), 2)

    team2_y_start = regions.TEAM2_ROWS[0]
    team2_y_end = regions.TEAM2_ROWS[-1] + regions.ROW_HEIGHT
    cv2.rectangle(result, (left_edge, team2_y_start), (right_edge, team2_y_end), (100, 100, 255), 2)

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "debug_edges.png"
    cv2.imwrite(str(output_path), result)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
