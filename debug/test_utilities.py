"""Shared utilities for debug/test scripts.

Provides common functionality for loading screenshots, calculating column positions,
and saving debug output.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Add src to path for imports
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
from src import regions


def get_screenshot_path(args: list[str] | None = None) -> Path | None:
    """Get screenshot path from command line arguments or find first available.

    Args:
        args: Command line arguments (defaults to sys.argv)

    Returns:
        Path to screenshot or None if not found
    """
    if args is None:
        args = sys.argv

    if len(args) > 1:
        screenshot_path = Path(args[1])
        if not screenshot_path.is_absolute():
            screenshot_path = SCREENSHOTS_DIR / screenshot_path
    else:
        images = list(SCREENSHOTS_DIR.glob("*.png")) + list(SCREENSHOTS_DIR.glob("*.jpg"))
        if not images:
            return None
        screenshot_path = sorted(images)[0]
        print(f"No image specified, using: {screenshot_path}")

    if not screenshot_path.exists():
        print(f"Screenshot not found: {screenshot_path}")
        return None

    return screenshot_path


def load_screenshot_with_columns(screenshot_path: Path) -> tuple[np.ndarray, dict[str, tuple[int, int]], int, int]:
    """Load a screenshot and calculate column positions.

    Args:
        screenshot_path: Path to the screenshot file

    Returns:
        Tuple of (image, columns, left_edge, right_edge)
    """
    image = load_image(screenshot_path)
    left_edge, right_edge = detect_scoreboard_edges(image)
    columns = calculate_column_positions(left_edge, right_edge)

    return image, columns, left_edge, right_edge


def create_output_dir(name: str) -> Path:
    """Create an output directory for test results.

    Args:
        name: Name of the output directory

    Returns:
        Path to the created directory
    """
    output_dir = OUTPUT_DIR / "test" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def print_header(title: str, width: int = 60) -> None:
    """Print a formatted header.

    Args:
        title: Header text
        width: Total width of the header line
    """
    print("=" * width)
    print(title)
    print("=" * width)


def print_section(title: str, width: int = 40) -> None:
    """Print a formatted section header.

    Args:
        title: Section text
        width: Total width of the separator line
    """
    print(f"\n{title}")
    print("-" * width)


# Re-export commonly used functions
__all__ = [
    "get_screenshot_path",
    "load_screenshot_with_columns",
    "create_output_dir",
    "print_header",
    "print_section",
    "load_image",
    "save_crop",
    "detect_scoreboard_edges",
    "calculate_column_positions",
    "crop_cell",
    "SCREENSHOTS_DIR",
    "OUTPUT_DIR",
    "TEMPLATES_DIR",
    "regions",
]
