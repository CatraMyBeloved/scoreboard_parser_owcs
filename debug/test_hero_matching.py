"""Test hero template matching.

Tests hero portrait recognition against sample crops from screenshots.
Verifies confidence thresholds and tests both team colors (cyan/yellow).

Usage:
    uv run python debug/test_hero_matching.py [screenshot_path]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from test_utilities import (
    get_screenshot_path,
    load_screenshot_with_columns,
    create_output_dir,
    print_header,
    print_section,
    crop_cell,
    save_crop,
)
from src.recognize import (
    match_hero,
    get_template_manager,
    HERO_CONFIDENCE_THRESHOLD,
)


def main() -> None:
    screenshot_path = get_screenshot_path()
    if screenshot_path is None:
        print("No screenshots found in screenshots/")
        print("Usage: uv run python debug/test_hero_matching.py [screenshot_path]")
        return

    print_header(f"HERO MATCHING TEST: {screenshot_path.name}")

    image, columns, left_edge, right_edge = load_screenshot_with_columns(screenshot_path)
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print(f"Scoreboard edges: left={left_edge}, right={right_edge}")

    templates = get_template_manager().load_templates("heroes_cropped")
    print(f"Loaded {len(templates)} hero templates")
    print(f"Confidence threshold: {HERO_CONFIDENCE_THRESHOLD}")

    output_dir = create_output_dir(f"hero_match_{screenshot_path.stem}")

    results = {"matched": 0, "unmatched": 0, "low_confidence": 0}

    for team in [1, 2]:
        team_name = "Team 1 (Cyan)" if team == 1 else "Team 2 (Yellow)"
        print_section(team_name)

        team_dir = output_dir / f"team{team}"
        team_dir.mkdir(exist_ok=True)

        for row in range(5):
            crop = crop_cell(image, team, row, "hero", columns)
            match = match_hero(crop, templates)

            save_crop(crop, team_dir / f"row{row}_hero.png")

            if match is None:
                status = "NO MATCH"
                results["unmatched"] += 1
                print(f"  Row {row}: {status} (below threshold)")
            else:
                if match.confidence >= HERO_CONFIDENCE_THRESHOLD:
                    status = "OK"
                    results["matched"] += 1
                else:
                    status = "LOW"
                    results["low_confidence"] += 1
                print(f"  Row {row}: {match.name:<15} (confidence: {match.confidence:.3f}) [{status}]")

    print_section("Summary")
    print(f"  Matched:        {results['matched']}")
    print(f"  Unmatched:      {results['unmatched']}")
    print(f"  Low confidence: {results['low_confidence']}")
    print(f"\nCrops saved to: {output_dir}")


if __name__ == "__main__":
    main()
