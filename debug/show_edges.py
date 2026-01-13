"""Debug script to visualize detected scoreboard edges.

Usage:
    uv run python debug/show_edges.py [screenshot_path]
"""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import regions
from src.utils import (
    OUTPUT_DIR,
    SCREENSHOTS_DIR,
    detect_scoreboard_edges,
    load_image,
)


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
    print("Detecting scoreboard edges...")
    try:
        left_edge, right_edge = detect_scoreboard_edges(image)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Detected edges: left={left_edge}, right={right_edge}")
    print(f"Scoreboard width: {right_edge - left_edge}px")

    # Draw detections
    result = image.copy()
    scan_y = regions.TEAM1_ROWS[0] - 5

    # Scan line
    cv2.line(result, (0, scan_y), (image.shape[1], scan_y), (0, 255, 255), 1)

    # Edge markers
    cv2.circle(result, (left_edge, scan_y), 5, (0, 255, 0), -1)
    cv2.circle(result, (right_edge, scan_y), 5, (0, 0, 255), -1)

    # Team 1 box
    team1_y_start = regions.TEAM1_ROWS[0]
    team1_y_end = regions.TEAM1_ROWS[-1] + regions.ROW_HEIGHT
    cv2.rectangle(result, (left_edge, team1_y_start), (right_edge, team1_y_end), (255, 100, 100), 2)

    # Team 2 box
    team2_y_start = regions.TEAM2_ROWS[0]
    team2_y_end = regions.TEAM2_ROWS[-1] + regions.ROW_HEIGHT
    cv2.rectangle(result, (left_edge, team2_y_start), (right_edge, team2_y_end), (100, 100, 255), 2)

    # Save result
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "debug_edges.png"
    cv2.imwrite(str(output_path), result)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
