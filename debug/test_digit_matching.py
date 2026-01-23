"""Test script for digit template matching.

Matches digit templates (0-9) against stat cell crops using sliding window
template matching. Outputs detected numbers and visualizations.

Usage:
    python debug/test_digit_matching.py [screenshot_path]
"""

import sys
from pathlib import Path
from dataclasses import dataclass

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
    TEMPLATES_DIR,
)


# Stat columns to test
STAT_COLUMNS = ["elims", "assists", "deaths", "damage", "healing", "mit"]


# -----------------------------------------------------------------------------
# Core Functions
# -----------------------------------------------------------------------------


def load_digit_templates() -> dict[str, np.ndarray]:
    """Load digit templates (0-9) from templates/digits/.

    Templates are already 4x scaled, so we only convert to binary.

    Returns:
        Dict mapping digit string ('0'-'9') to binary template.
    """
    templates = {}
    digits_dir = TEMPLATES_DIR / "digits"

    for digit in range(10):
        path = digits_dir / f"{digit}.png"
        if path.exists():
            img = cv2.imread(str(path))
            if img is not None:
                # Templates are already scaled, just convert to binary
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                templates[str(digit)] = binary

    return templates


def preprocess(crop: np.ndarray) -> np.ndarray:
    """Preprocess crop for template matching.

    Pipeline: 4x scale -> grayscale -> CLAHE -> Otsu threshold

    Args:
        crop: BGR image

    Returns:
        Binary image (white digits on black background)
    """
    # Scale 4x
    scaled = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # Grayscale
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Otsu threshold (NOT inverted - keeps white digits on black background)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


@dataclass
class DigitMatch:
    """A detected digit match."""
    digit: str
    x: int
    y: int
    confidence: float
    width: int
    height: int


def match_digits(
    image: np.ndarray,
    templates: dict[str, np.ndarray],
    threshold: float = 0.7,
    debug: bool = False,
) -> list[DigitMatch]:
    """Find all digit matches in an image using sliding window template matching.

    Args:
        image: Preprocessed binary image
        templates: Dict of digit -> binary template
        threshold: Minimum confidence for a match
        debug: If True, print max confidence per digit

    Returns:
        List of DigitMatch objects (unsorted, may have overlaps)
    """
    matches = []

    for digit, template in templates.items():
        th, tw = template.shape[:2]
        ih, iw = image.shape[:2]

        # Skip if template is larger than image
        if th > ih or tw > iw:
            if debug:
                print(f"      Digit {digit}: template {tw}x{th} > image {iw}x{ih}, skipped")
            continue

        # Template matching
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        max_val = float(result.max())

        if debug:
            print(f"      Digit {digit}: max_conf={max_val:.3f}")

        # Find all locations above threshold
        locations = np.where(result >= threshold)

        for y, x in zip(*locations):
            confidence = float(result[y, x])
            matches.append(DigitMatch(
                digit=digit,
                x=int(x),
                y=int(y),
                confidence=confidence,
                width=tw,
                height=th,
            ))

    return matches


def non_max_suppression(
    matches: list[DigitMatch],
    min_distance: int = 20,
) -> list[DigitMatch]:
    """Remove overlapping matches, keeping highest confidence.

    Args:
        matches: List of digit matches
        min_distance: Minimum X distance between matches

    Returns:
        Filtered list with overlaps removed
    """
    if not matches:
        return []

    # Sort by confidence (highest first)
    sorted_matches = sorted(matches, key=lambda m: -m.confidence)

    kept = []
    for match in sorted_matches:
        # Check if this match overlaps with any kept match
        overlaps = False
        for kept_match in kept:
            if abs(match.x - kept_match.x) < min_distance:
                overlaps = True
                break

        if not overlaps:
            kept.append(match)

    return kept


def matches_to_number(matches: list[DigitMatch]) -> str:
    """Convert sorted matches to a number string.

    Args:
        matches: List of DigitMatch objects

    Returns:
        String representation of the detected number
    """
    # Sort by X position (left to right)
    sorted_matches = sorted(matches, key=lambda m: m.x)
    return ''.join(m.digit for m in sorted_matches)


def visualize_matches(
    image: np.ndarray,
    matches: list[DigitMatch],
) -> np.ndarray:
    """Draw bounding boxes and labels on the image.

    Args:
        image: Grayscale or binary image
        matches: List of digit matches

    Returns:
        BGR image with annotations
    """
    # Convert to BGR if grayscale
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    # Sort matches by X for consistent visualization
    sorted_matches = sorted(matches, key=lambda m: m.x)

    for match in sorted_matches:
        # Draw bounding box at actual match position
        x1 = match.x
        x2 = match.x + match.width
        y1 = match.y
        y2 = match.y + match.height

        # Green box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label with digit and confidence
        label = f"{match.digit} ({match.confidence:.2f})"
        cv2.putText(
            vis, label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 1,
        )

    return vis


# -----------------------------------------------------------------------------
# Main Processing
# -----------------------------------------------------------------------------


def process_stat_cell(
    crop: np.ndarray,
    templates: dict[str, np.ndarray],
    threshold: float = 0.7,
    debug: bool = False,
) -> tuple[str, list[DigitMatch], np.ndarray]:
    """Process a single stat cell and detect the number.

    Args:
        crop: BGR image of the stat cell
        templates: Digit templates
        threshold: Match confidence threshold
        debug: If True, print debug info

    Returns:
        Tuple of (detected_number, matches, visualization)
    """
    # Preprocess the crop
    processed = preprocess(crop)

    if debug:
        print(f"    Processed crop size: {processed.shape[1]}x{processed.shape[0]}")

    # Find digit matches
    raw_matches = match_digits(processed, templates, threshold, debug=debug)

    if debug:
        print(f"    Raw matches: {len(raw_matches)}")

    # Apply non-max suppression
    filtered_matches = non_max_suppression(raw_matches)

    # Convert to number string
    number = matches_to_number(filtered_matches)

    # Create visualization
    vis = visualize_matches(processed, filtered_matches)

    return number, filtered_matches, vis


def process_frame(screenshot_path: Path) -> None:
    """Process a screenshot and test digit matching on all stat cells."""
    print(f"Processing: {screenshot_path.name}")
    print("=" * 60)

    # Load digit templates
    templates = load_digit_templates()
    if not templates:
        print("ERROR: No digit templates found in templates/digits/")
        return

    print(f"Loaded {len(templates)} digit templates")

    # Print template sizes
    for digit, template in sorted(templates.items()):
        h, w = template.shape[:2]
        print(f"  {digit}: {w}x{h}")

    # Load image
    image = load_image(screenshot_path)
    print(f"\nImage size: {image.shape[1]}x{image.shape[0]}")

    # Detect edges and calculate columns
    left_edge, right_edge = detect_scoreboard_edges(image)
    columns = calculate_column_positions(left_edge, right_edge)
    print(f"Scoreboard edges: left={left_edge}, right={right_edge}")

    # Create output directory
    frame_name = screenshot_path.stem.replace(" ", "_")
    output_dir = OUTPUT_DIR / "test" / f"digit_match_{frame_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}\n")

    # Process all stat cells
    for team in (1, 2):
        print(f"Team {team}:")
        team_dir = output_dir / f"team{team}"
        team_dir.mkdir(exist_ok=True)

        for row in range(5):
            row_results = []

            for col_name in STAT_COLUMNS:
                if col_name not in columns:
                    continue

                # Crop cell
                crop = crop_cell(image, team, row, col_name, columns)

                # Process and detect number
                number, matches, vis = process_stat_cell(crop, templates)

                # Save visualization
                vis_path = team_dir / f"p{row + 1}_{col_name}.png"
                save_crop(vis, vis_path)

                row_results.append(f"{col_name}={number or '?'}")

            print(f"  Player {row + 1}: {', '.join(row_results)}")

    print(f"\nDone! Visualizations saved to: {output_dir}")


def main():
    """Run digit matching test on a screenshot."""
    if len(sys.argv) > 1:
        screenshot_path = Path(sys.argv[1])
        if not screenshot_path.is_absolute():
            screenshot_path = SCREENSHOTS_DIR / screenshot_path
    else:
        # Find first PNG in screenshots folder
        screenshots = sorted(SCREENSHOTS_DIR.glob("*.png"))

        if not screenshots:
            print("No screenshots found in screenshots/")
            print("Usage: python debug/test_digit_matching.py [screenshot_path]")
            return

        screenshot_path = screenshots[0]

    if not screenshot_path.exists():
        print(f"Screenshot not found: {screenshot_path}")
        return

    process_frame(screenshot_path)


if __name__ == "__main__":
    main()
