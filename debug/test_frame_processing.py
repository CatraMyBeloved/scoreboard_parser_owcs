"""Test full frame processing with all recognition features.

Usage:
    uv run python debug/test_frame_processing.py [screenshot_path]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recognize import process_frame
from src.utils import SCREENSHOTS_DIR, load_image


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

    # Load image
    print(f"Loading {screenshot_path}...")
    image = load_image(screenshot_path)

    # Process frame
    print("Processing frame (this may take a moment for OCR initialization)...")
    players = process_frame(image)

    # Display results
    print("\n" + "=" * 80)
    print("FRAME PROCESSING RESULTS")
    print("=" * 80)

    for team in [1, 2]:
        team_name = "Team 1 (Blue)" if team == 1 else "Team 2 (Yellow)"
        print(f"\n{team_name}")
        print("-" * 40)

        team_players = [p for p in players if p.team == team]
        for player in team_players:
            # Format stats
            stats = []
            if player.elims is not None:
                stats.append(f"E:{player.elims}")
            if player.assists is not None:
                stats.append(f"A:{player.assists}")
            if player.deaths is not None:
                stats.append(f"D:{player.deaths}")
            if player.damage is not None:
                stats.append(f"Dmg:{player.damage}")
            if player.healing is not None:
                stats.append(f"Heal:{player.healing}")
            if player.mit is not None:
                stats.append(f"Mit:{player.mit}")

            stats_str = " | ".join(stats) if stats else "No stats"
            role_str = player.role or "?"
            name_str = player.name or "Unknown"

            print(f"  [{role_str:7s}] {name_str:15s} | {stats_str}")


if __name__ == "__main__":
    main()
