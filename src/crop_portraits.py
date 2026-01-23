"""Crop hero portraits from full team screenshots.

Extracts hero portraits from both teams using existing crop infrastructure.
Saves with numbered filenames for manual labeling.
"""

from pathlib import Path

from src.utils import (load_image, save_crop, crop_cell,
                     detect_scoreboard_edges, calculate_column_positions, SCREENSHOTS_DIR, TEMPLATES_DIR)


def crop_all_hero_portraits(output_dir: Path) -> int:
    """Crop hero portraits from all 'full team *.png' screenshots.

    Returns the total number of portraits cropped.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    screenshots = sorted(SCREENSHOTS_DIR.glob("full team *.png"))

    if not screenshots:
        print("No 'full team *.png' files found in screenshots/")
        return 0

    print(f"Found {len(screenshots)} screenshots")
    print(f"Output directory: {output_dir}")
    print()

    idx = 1

    for screenshot_path in screenshots:
        print(f"Processing: {screenshot_path.name}")

        image = load_image(screenshot_path)
        left_edge, right_edge = detect_scoreboard_edges(image)
        columns = calculate_column_positions(left_edge, right_edge)

        # Crop team 1 (cyan) hero portraits
        for row in range(5):
            crop = crop_cell(image, team=1, row=row, column="hero", columns=columns)
            output_path = output_dir / f"cyan_{idx:03d}.png"
            save_crop(crop, output_path)
            print(f"  Saved: {output_path.name}")
            idx += 1

        # Crop team 2 (yellow) hero portraits
        for row in range(5):
            crop = crop_cell(image, team=2, row=row, column="hero", columns=columns)
            output_path = output_dir / f"yellow_{idx:03d}.png"
            save_crop(crop, output_path)
            print(f"  Saved: {output_path.name}")
            idx += 1

    return idx - 1


def main():
    output_dir = TEMPLATES_DIR / "heroes_cropped"
    total = crop_all_hero_portraits(output_dir)
    print()
    print(f"Done! Cropped {total} portraits to {output_dir}")


if __name__ == "__main__":
    main()
