"""Test ultimate status detection.

Tests both checkmark detection (ready state) and percentage OCR (charging state).
Verifies edge cases (0%, 100%) and confidence thresholds.

Usage:
    uv run python debug/test_ultimate_detection.py [screenshot_path]
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
    detect_ult_status,
    detect_ultimate_ready_checkmark,
    read_ultimate_charge_percentage,
    get_template_manager,
    ULT_READY_CONFIDENCE_THRESHOLD,
    extract_by_saturation,
)


def main() -> None:
    screenshot_path = get_screenshot_path()
    if screenshot_path is None:
        print("No screenshots found in screenshots/")
        print("Usage: uv run python debug/test_ultimate_detection.py [screenshot_path]")
        return

    print_header(f"ULTIMATE DETECTION TEST: {screenshot_path.name}")

    image, columns, left_edge, right_edge = load_screenshot_with_columns(screenshot_path)
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print(f"Scoreboard edges: left={left_edge}, right={right_edge}")

    templates = get_template_manager().load_templates("ult")
    print(f"Loaded {len(templates)} ult templates")
    print(f"Ready threshold: {ULT_READY_CONFIDENCE_THRESHOLD}")

    output_dir = create_output_dir(f"ult_detect_{screenshot_path.stem}")

    results = {"ready": 0, "charging": 0, "unknown": 0}

    for team in [1, 2]:
        team_name = "Team 1 (Cyan)" if team == 1 else "Team 2 (Yellow)"
        print_section(team_name)

        team_dir = output_dir / f"team{team}"
        team_dir.mkdir(exist_ok=True)

        for row in range(5):
            crop = crop_cell(image, team, row, "ult", columns)
            is_ready, charge = detect_ult_status(crop, templates)

            save_crop(crop, team_dir / f"row{row}_ult.png")
            save_crop(extract_by_saturation(crop), team_dir / f"row{row}_ult_binary.png")

            if is_ready:
                status = "READY (checkmark)"
                results["ready"] += 1
            elif charge is not None:
                status = f"CHARGING: {charge}%"
                results["charging"] += 1
            else:
                status = "UNKNOWN"
                results["unknown"] += 1

            checkmark_detected = detect_ultimate_ready_checkmark(crop, templates)
            ocr_result = read_ultimate_charge_percentage(crop)

            print(f"  Row {row}: {status}")
            print(f"           Checkmark: {checkmark_detected}, OCR: {ocr_result}")

    print_section("Summary")
    print(f"  Ready:    {results['ready']}")
    print(f"  Charging: {results['charging']}")
    print(f"  Unknown:  {results['unknown']}")
    print(f"\nCrops saved to: {output_dir}")


if __name__ == "__main__":
    main()
