"""Test role template matching across both teams.

Usage:
    uv run python debug/test_role_matching.py [screenshot_path]
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    SCREENSHOTS_DIR,
    TEMPLATES_DIR,
    crop_cell,
    detect_scoreboard_edges,
    calculate_column_positions,
    load_image,
)


def load_role_templates() -> dict[str, np.ndarray]:
    """Load role templates as BGR images."""
    templates = {}
    roles_dir = TEMPLATES_DIR / "roles"
    for role in ["tank", "dps", "support"]:
        path = roles_dir / f"{role}.png"
        if path.exists():
            templates[role] = cv2.imread(str(path))
    return templates


def extract_icon(image_bgr: np.ndarray) -> np.ndarray:
    """Extract white icon from background using saturation.

    White has low saturation, colored backgrounds (blue/yellow) have high saturation.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    # Low saturation = white icon, high saturation = colored background
    # Invert so white icon becomes white (255) in output
    _, binary = cv2.threshold(saturation, 50, 255, cv2.THRESH_BINARY_INV)
    return binary


def match_role(crop: np.ndarray, templates_bgr: dict[str, np.ndarray]) -> tuple[str, float]:
    """Match a role crop against templates.

    Returns (role_name, confidence) for best match.
    """
    crop_binary = extract_icon(crop)

    best_role = None
    best_score = -1

    for role, template_bgr in templates_bgr.items():
        # Extract icon from template too
        template_binary = extract_icon(template_bgr)

        # Resize template to match crop if needed
        if template_binary.shape != crop_binary.shape:
            template_resized = cv2.resize(template_binary, (crop_binary.shape[1], crop_binary.shape[0]))
        else:
            template_resized = template_binary

        # Template matching on binary images
        result = cv2.matchTemplate(crop_binary, template_resized, cv2.TM_CCOEFF_NORMED)
        score = result[0, 0]  # Single value since same size

        if score > best_score:
            best_score = score
            best_role = role

    return best_role, best_score


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

    # Load image and templates
    print(f"Loading {screenshot_path}...")
    image = load_image(screenshot_path)
    templates = load_role_templates()
    print(f"Loaded {len(templates)} role templates: {list(templates.keys())}")

    # Detect edges and calculate columns once
    left_edge, right_edge = detect_scoreboard_edges(image)
    columns = calculate_column_positions(left_edge, right_edge)

    # Test matching on all players
    print("\nRole matching results:")
    print("-" * 50)

    for team in [1, 2]:
        team_name = "Team 1 (Blue)" if team == 1 else "Team 2 (Yellow)"
        print(f"\n{team_name}:")

        for row in range(5):
            crop = crop_cell(image, team, row, "role", columns)
            role, confidence = match_role(crop, templates)
            status = "OK" if confidence > 0.5 else "LOW"
            print(f"  Row {row}: {role:8s} (confidence: {confidence:.3f}) [{status}]")


if __name__ == "__main__":
    main()
