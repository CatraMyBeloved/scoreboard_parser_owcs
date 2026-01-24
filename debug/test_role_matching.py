"""Test role template matching across both teams.

Usage:
    uv run python debug/test_role_matching.py [screenshot_path]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from test_utilities import (
    get_screenshot_path,
    load_screenshot_with_columns,
    print_header,
    print_section,
    crop_cell,
)
from src.recognize import match_role, get_template_manager


def main() -> None:
    screenshot_path = get_screenshot_path()
    if screenshot_path is None:
        print("No screenshots found in screenshots/")
        print("Usage: uv run python debug/test_role_matching.py [screenshot_path]")
        return

    print_header(f"ROLE MATCHING TEST: {screenshot_path.name}")

    image, columns, left_edge, right_edge = load_screenshot_with_columns(screenshot_path)
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print(f"Scoreboard edges: left={left_edge}, right={right_edge}")

    templates = get_template_manager().load_templates("roles")
    print(f"Loaded {len(templates)} role templates: {list(templates.keys())}")

    for team in [1, 2]:
        team_name = "Team 1 (Cyan)" if team == 1 else "Team 2 (Yellow)"
        print_section(team_name)

        for row in range(5):
            crop = crop_cell(image, team, row, "role", columns)
            role_match = match_role(crop, templates)
            status = "OK" if role_match.confidence > 0.5 else "LOW"
            print(f"  Row {row}: {role_match.name:8s} (confidence: {role_match.confidence:.3f}) [{status}]")


if __name__ == "__main__":
    main()
