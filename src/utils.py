"""Shared utility functions for scoreboard parsing."""

from pathlib import Path

import cv2
import numpy as np

from . import regions

PROJECT_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
OUTPUT_DIR = PROJECT_ROOT / "output"
CROPS_DIR = OUTPUT_DIR / "crops"


def ensure_dirs() -> None:
    """Ensure required directories exist."""
    SCREENSHOTS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    CROPS_DIR.mkdir(exist_ok=True)


def load_image(path: Path) -> np.ndarray:
    """Load image as BGR numpy array (OpenCV format)."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return img


def save_crop(crop: np.ndarray, path: Path) -> None:
    """Save a cropped image to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop)


# --- Edge Detection ---


def detect_scoreboard_edges(
    image: np.ndarray, threshold: int = 40
) -> tuple[int, int]:
    """Detect left and right edges of the scoreboard.

    Scans horizontally just above the first row to find brightness transitions.

    Args:
        image: Full screenshot as BGR numpy array
        threshold: Gradient threshold for edge detection

    Returns:
        (left_x, right_x) pixel coordinates

    Raises:
        ValueError: If edges cannot be detected
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scan_y = regions.TEAM1_ROWS[0] - 5
    row = gray[scan_y, :]

    # Compute gradient (difference between adjacent pixels)
    gradient = np.diff(row.astype(np.int16))

    # Find left edge: large positive gradient (dark -> bright)
    left_candidates = np.where(gradient > threshold)[0]
    if len(left_candidates) == 0:
        raise ValueError("Could not detect left edge of scoreboard")
    left_x = int(left_candidates[0])

    # Find right edge: large negative gradient (bright -> dark)
    right_candidates = np.where(gradient < -threshold)[0]
    if len(right_candidates) == 0:
        raise ValueError("Could not detect right edge of scoreboard")
    right_x = int(right_candidates[-1])

    return left_x, right_x


# --- Column Position Calculation ---


def calculate_column_positions(
    left_edge: int, right_edge: int
) -> dict[str, tuple[int, int]]:
    """Calculate column X positions based on detected edges and known widths.

    Args:
        left_edge: Left edge X coordinate of scoreboard
        right_edge: Right edge X coordinate of scoreboard

    Returns:
        Dict mapping column name to (x_start, x_end) tuple
    """
    columns = {}

    # Left columns (anchored from left edge)
    x = left_edge
    for col_name in regions.LEFT_COLUMNS_ORDER:
        width = regions.LEFT_COLUMNS[col_name]
        columns[col_name] = (x, x + width)
        x += width

    # Name section (fixed width, right of ult)
    name_start = left_edge + regions.LEFT_TOTAL
    name_end = name_start + regions.NAME_WIDTH
    columns["name"] = (name_start, name_end)

    # Perks section (variable width, fills gap between name and stats)
    perks_start = name_end
    perks_end = right_edge - regions.REPORT_BUTTON_WIDTH - regions.RIGHT_TOTAL
    columns["perks"] = (perks_start, perks_end)

    # Right columns (anchored from right edge, minus report button)
    x = right_edge - regions.REPORT_BUTTON_WIDTH
    for col_name in reversed(regions.RIGHT_COLUMNS_ORDER):
        width = regions.RIGHT_COLUMNS[col_name]
        columns[col_name] = (x - width, x)
        x -= width

    return columns


# --- Cropping Functions ---


# Column names available for cropping
ALL_COLUMNS = (
    regions.LEFT_COLUMNS_ORDER + ["name", "perks"] + regions.RIGHT_COLUMNS_ORDER
)


def crop_cell(
    image: np.ndarray,
    team: int,
    row: int,
    column: str,
    columns: dict[str, tuple[int, int]] | None = None,
) -> np.ndarray:
    """Crop a single cell from the scoreboard.

    Args:
        image: Full screenshot as numpy array
        team: 1 or 2
        row: 0-4 (player index within team)
        column: Column name (e.g., "hero", "elims", "name")
        columns: Pre-calculated column positions (optional, will detect if not provided)

    Returns:
        Cropped region as numpy array
    """
    if team not in (1, 2):
        raise ValueError(f"team must be 1 or 2, got {team}")
    if row not in range(5):
        raise ValueError(f"row must be 0-4, got {row}")

    # Calculate column positions if not provided
    if columns is None:
        left_edge, right_edge = detect_scoreboard_edges(image)
        columns = calculate_column_positions(left_edge, right_edge)

    if column not in columns:
        raise ValueError(f"Unknown column: {column}. Available: {list(columns.keys())}")

    # Get Y position from team rows
    row_y = regions.TEAM1_ROWS[row] if team == 1 else regions.TEAM2_ROWS[row]

    # Get X range from calculated columns
    x_start, x_end = columns[column]

    # Crop: image[y:y+h, x:x+w]
    return image[row_y : row_y + regions.ROW_HEIGHT, x_start:x_end]


def crop_all_cells(
    image: np.ndarray, skip_perks: bool = True
) -> dict[str, np.ndarray]:
    """Crop all cells from a screenshot.

    Args:
        image: Full screenshot as numpy array
        skip_perks: Whether to skip the perks column (default True)

    Returns:
        Dict keyed by "team{t}_row{r}_{column}" -> cropped image
    """
    # Detect edges once for all crops
    left_edge, right_edge = detect_scoreboard_edges(image)
    columns = calculate_column_positions(left_edge, right_edge)

    # Determine which columns to crop
    columns_to_crop = list(columns.keys())
    if skip_perks and "perks" in columns_to_crop:
        columns_to_crop.remove("perks")

    crops = {}
    for team in (1, 2):
        for row in range(5):
            for column in columns_to_crop:
                key = f"team{team}_row{row}_{column}"
                crops[key] = crop_cell(image, team, row, column, columns)

    return crops
