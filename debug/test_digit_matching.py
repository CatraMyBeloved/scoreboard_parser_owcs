"""Test digit template matching.

Matches digit templates (0-9) against stat cell crops using sliding window
template matching. Outputs detected numbers and visualizations.

Usage:
    uv run python debug/test_digit_matching.py [screenshot_path]
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from test_utilities import (
    get_screenshot_path,
    load_screenshot_with_columns,
    create_output_dir,
    print_header,
    crop_cell,
    save_crop,
)
from src.digit_matcher import (
    get_digit_matcher,
    _preprocess,
    _match_digits,
    _non_max_suppression,
    _matches_to_number,
    DigitMatch,
)


STAT_COLUMNS = ["elims", "assists", "deaths", "damage", "healing", "mit"]


def visualize_matches(image: np.ndarray, matches: list[DigitMatch]) -> np.ndarray:
    """Draw bounding boxes and labels on the image.

    Args:
        image: Grayscale or binary image
        matches: List of digit matches

    Returns:
        BGR image with annotations
    """
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    sorted_matches = sorted(matches, key=lambda m: m.x)

    for match in sorted_matches:
        x1, x2 = match.x, match.x + match.width
        y1, y2 = match.y, match.y + match.height

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{match.digit} ({match.confidence:.2f})"
        cv2.putText(vis, label, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return vis


def process_stat_cell(
    crop: np.ndarray,
    templates: dict[str, np.ndarray],
    threshold: float = 0.7,
) -> tuple[str, list[DigitMatch], np.ndarray]:
    """Process a single stat cell and detect the number.

    Args:
        crop: BGR image of the stat cell
        templates: Digit templates
        threshold: Match confidence threshold

    Returns:
        Tuple of (detected_number, matches, visualization)
    """
    processed = _preprocess(crop)
    raw_matches = _match_digits(processed, templates, threshold)
    filtered_matches = _non_max_suppression(raw_matches)
    number = _matches_to_number(filtered_matches)
    vis = visualize_matches(processed, filtered_matches)
    return number, filtered_matches, vis


def main() -> None:
    screenshot_path = get_screenshot_path()
    if screenshot_path is None:
        print("No screenshots found in screenshots/")
        print("Usage: uv run python debug/test_digit_matching.py [screenshot_path]")
        return

    print_header(f"DIGIT MATCHING TEST: {screenshot_path.name}")

    matcher = get_digit_matcher()
    templates = matcher.templates
    if not templates:
        print("ERROR: No digit templates found in templates/digits/")
        return

    print(f"Loaded {len(templates)} digit templates")
    for digit, template in sorted(templates.items()):
        height, width = template.shape[:2]
        print(f"  {digit}: {width}x{height}")

    image, columns, left_edge, right_edge = load_screenshot_with_columns(screenshot_path)
    print(f"\nImage size: {image.shape[1]}x{image.shape[0]}")
    print(f"Scoreboard edges: left={left_edge}, right={right_edge}")

    output_dir = create_output_dir(f"digit_match_{screenshot_path.stem}")
    print(f"Output: {output_dir}\n")

    for team in (1, 2):
        print(f"Team {team}:")
        team_dir = output_dir / f"team{team}"
        team_dir.mkdir(exist_ok=True)

        for row in range(5):
            row_results = []

            for col_name in STAT_COLUMNS:
                if col_name not in columns:
                    continue

                crop = crop_cell(image, team, row, col_name, columns)
                number, matches, vis = process_stat_cell(crop, templates)

                vis_path = team_dir / f"p{row + 1}_{col_name}.png"
                save_crop(vis, vis_path)

                row_results.append(f"{col_name}={number or '?'}")

            print(f"  Player {row + 1}: {', '.join(row_results)}")

    print(f"\nDone! Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
